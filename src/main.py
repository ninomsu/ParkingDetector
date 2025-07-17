import cv2
import torch
import time
import os
import sys
from pushover import Client
import xml.etree.ElementTree as ET
import logging
from datetime import datetime
import atexit
from functools import wraps
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.metrics import set_meter_provider
from opentelemetry.metrics import Observation
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk._logs import LoggingHandler, LoggerProvider
from opentelemetry._logs import set_logger_provider, get_logger
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# === YOLO model and constants ===
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', trust_repo=True)
model.conf = 0.3

EMPTY = "Empty"
TAKEN = "Taken"
ERROR = "Error"
DEBUG = "debug"
TEST = "test"
RELEASE = "release"

MODE = DEBUG
cap = None
_meter = None
_counters = {}
_gauges = {}
_confidences = {}

# === Logging Setup ===
class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

def log(msg):
    if MODE == DEBUG:
        print(msg)
    logging.info(msg)

def otel_init(rois):
    global _meter, _counters, _gauges

    # === Logging Setup ===
    log_filename = f"parking_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for h in list(logger.handlers):
        logger.removeHandler(h)

    file_handler = FlushFileHandler(log_filename, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    resource = Resource.create({
    "service.name": "parking-detector-service",
    "service.version": "1.0.0",
    "host.name": os.uname().nodename,
    "os.type": sys.platform,
    })
    # === OpenTelemetry Logging ===
    log_exporter = OTLPLogExporter()
    log_provider = LoggerProvider(resource=resource)
    log_processor = BatchLogRecordProcessor(log_exporter)
    log_provider.add_log_record_processor(log_processor)
    set_logger_provider(log_provider)
    otel_log_handler = LoggingHandler(level=logging.INFO, logger_provider=log_provider)
    logger.addHandler(otel_log_handler)

    # === OpenTelemetry Metrics ===
    metric_exporter = OTLPMetricExporter()
    metric_reader = PeriodicExportingMetricReader(metric_exporter)
    metric_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    set_meter_provider(metric_provider)
    _meter = metric_provider.get_meter("parking-detector")

    _counters["taken"] = _meter.create_counter("spots_taken")
    _counters["empty"] = _meter.create_counter("spots_empty")

    _gauges["taken"] = {}
    _gauges["empty"] = {}
    for roi in rois:
        label = roi['label']
        _gauges["taken"][label] = _meter.create_up_down_counter(f"spots_taken_gauge_{label}")
        _gauges["empty"][label] = _meter.create_up_down_counter(f"spots_empty_gauge_{label}")
    
    def confidence_callback(options):
        for label, value in _confidences.items():
            yield Observation(value, {"spot": label})

    _meter.create_observable_gauge(
        "last_detection_confidence",
        callbacks=[confidence_callback],
        unit="1",
        description="Most recent detection confidence for each intersecting car"
    )
    
    # === OpenTelemetry Tracing ===
    trace_exporter = OTLPSpanExporter()
    tracer_provider = TracerProvider(resource=resource)
    span_processor = BatchSpanProcessor(trace_exporter)
    tracer_provider.add_span_processor(span_processor)
    trace.set_tracer_provider(tracer_provider)
    _tracer = trace.get_tracer("parking-detector")

    atexit.register(logging.shutdown)

def get_counter(name):
    return _counters[name]

def get_gauge(category, label):
    return _gauges[category][label]

def traced(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if MODE != DEBUG:
            return func(*args, **kwargs)
        tracer = trace.get_tracer("parking-detector")
        with tracer.start_as_current_span(func.__name__):
            return func(*args, **kwargs)
    return wrapper

# === Notification (optional) ===
@traced
def load_credentials(xml_path):
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Credential file not found: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    api_token = root.find('api-token').text.strip()
    user_key = root.find('user-key').text.strip()
    return api_token, user_key

@traced
def send_notification(spot, state, header):
    api_token, user_key = load_credentials("./credentials.xml")
    client = Client(user_key, api_token=api_token)
    client.send_message(f"{header} Spot '{spot}' is {state}", title=f"Parking Alert – {spot}")

# === Utility Functions ===
@traced
def boxes_intersect(boxA, boxB):
    ax1, ay1, aw, ah = boxA
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = boxB
    bx2, by2 = bx1 + bw, by1 + bh
    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

def is_within_active_hours(start, end):
    now = datetime.now().time()
    return datetime.strptime(str(start) + ":00", "%H:%M").time() <= now <= datetime.strptime(str(end) + ":00", "%H:%M").time()

# === Drawing on Frame ===
@traced
def draw_frame(frame, states, detections, rois, conf_threshold):
    for *box, conf, cls in detections:
        if int(cls) in [2, 7] and float(conf) >= conf_threshold:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            label_text = f"{int(cls)} {conf:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    for roi in rois:
        x, y, w, h = roi['coords']
        state = states.get(roi['label'], EMPTY)
        color = (0, 0, 255) if state == TAKEN else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{roi['label']}: {state}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

@traced
def load_config(xml_path):
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Config file not found: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    mode = root.find("mode").text.strip()
    samples = int(root.find("samples").text)
    confidence = float(root.find("confidence").text)
    active_sleep = int(root.find("active_sleep_s").text)
    passive_sleep = int(root.find("passive_sleep_s").text)
    start_hour = int(root.find("start_hour").text)
    end_hour = int(root.find("end_hour").text)
    
    return {
        "mode": mode.lower(),
        "samples": int(samples),
        "confidence": float(confidence),
        "active_sleep_s": int(active_sleep),
        "passive_sleep_s": int(passive_sleep),
        "start_hour": int(start_hour),
        "end_hour": int(end_hour),
    }

@traced
def load_rois(xml_path):
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"ROI XML file not found: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rois = []
    for roi_elem in root.findall("roi"):
        label = roi_elem.find("label").text.strip()
        x = int(roi_elem.find("x").text)
        y = int(roi_elem.find("y").text)
        w = int(roi_elem.find("width").text)
        h = int(roi_elem.find("height").text)
        rois.append({"label": label, "coords": (x, y, w, h)})
    return rois

@traced
def detect_and_vote(cap, rois, model, samples, conf_threshold):
    taken_counts = {roi['label']: 0 for roi in rois}
    empty_counts = {roi['label']: 0 for roi in rois}
    last_frame = None

    for _ in range(samples):
        ret, frame = cap.read()
        if not ret:
            log("[ERROR] Camera read failed")
            break

        last_frame = frame.copy()
        results = model(frame)
        detections = results.xyxy[0]

        for roi in rois:
            label = roi['label']
            coords = roi['coords']
            spot_taken = False
            for *box, conf, cls in detections:
                if int(cls) in [2, 7] and float(conf) >= conf_threshold:
                    x1, y1, x2, y2 = map(int, box)
                    car_box = (x1, y1, x2 - x1, y2 - y1)
                    if boxes_intersect(car_box, coords):
                        conf_val = float(conf)
                        log(f"{label}: Detected class {int(cls)} with conf {conf_val:.2f}")
                        _confidences[label] = conf_val
                        spot_taken = True
                        break
            if spot_taken:
                taken_counts[label] += 1
            else:
                empty_counts[label] += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            logging.shutdown()
            sys.exit(0)
        time.sleep(1)

    # Update gauges
    for label in taken_counts:
        _gauges["taken"][label].add(taken_counts[label], {"spot": label})
        _gauges["empty"][label].add(empty_counts[label], {"spot": label})

    return taken_counts, empty_counts, last_frame

@traced
def evaluate_states(rois, taken_counts, empty_counts):
    states = {}
    for roi in rois:
        label = roi['label']
        taken = taken_counts[label]
        empty = empty_counts[label]
        if taken > empty:
            states[label] = TAKEN
            get_counter("taken").add(1, {"spot": label})
        elif empty > taken:
            states[label] = EMPTY
            get_counter("empty").add(1, {"spot": label})
        else:
            states[label] = TAKEN  # default to worst case
        log(f"ROI {label}: Taken={taken}, Empty={empty}")
    return states

@traced
def handle_notifications_and_snapshots(rois, prev_states, current_states, base_frame, cycle_reset, conf_threshold):
    detections = model(base_frame).xyxy[0]
    annotated = draw_frame(base_frame.copy(), current_states, detections, rois, conf_threshold)
    if MODE == DEBUG:
        cv2.imshow("Parking Detection", annotated)
    for roi in rois:
        label = roi['label']
        if current_states[label] != prev_states[label] and cycle_reset:
            if MODE == DEBUG:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"snapshots/{label}_{current_states[label]}_{ts}.png"
                cv2.imwrite(fname, annotated)
                log(f"Snapshot saved for {label}: {fname}")
            send_notification(label, current_states[label], "Update:")

        if not cycle_reset:
            if MODE == DEBUG:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"snapshots/{label}_initial_{ts}.png"
                cv2.imwrite(fname, annotated)
            send_notification(label, current_states[label], "Initial:")
    return annotated

# === Main Execution ===
def main():
    # Create snapshots directory
    os.makedirs("snapshots", exist_ok=True)

    config = load_config("./config.xml")

    global MODE
    MODE = config.get("mode", "debug")
    samples = config.get("samples", 60)
    conf_threshold = config.get("confidence", 0.40)
    sleep_active = config.get("active_sleep_s", 180)
    sleep_passive = config.get("passive_sleep_s", 300)
    start = config.get("start_hour", 11)
    end = config.get("end_hour", 21)

    if start >= 24 or start < 0:
        start = 11
        log("Invalid start time given, start at " + str(start) + ":00")
    
    if end >= 24 or end < 0:
        end = 21
        log("Invalid end time given, end at " + str(end) + ":00")

    if start >= end:
        log("Invalid time given, start >= end")
        sys.exit(0)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    log("Warming up camera...")
    time.sleep(2)

    rois = load_rois("./rois.xml")
    prev_states = {roi['label']: EMPTY for roi in rois}
    cycle_reset = False
    
    otel_init(rois)
    
    while True:
        if not is_within_active_hours(start, end):
            log("Not within active hours")
            time.sleep(sleep_passive)
            if cycle_reset:
                cycle_reset = False
            continue

        taken_counts, empty_counts, base_frame = detect_and_vote(cap, rois, model, samples, conf_threshold)
        current_states = evaluate_states(rois, taken_counts, empty_counts)
        handle_notifications_and_snapshots(rois, prev_states, current_states, base_frame, cycle_reset, conf_threshold)
        prev_states = current_states
        if not cycle_reset:
            cycle_reset = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(sleep_active)

    cap.release()
    cv2.destroyAllWindows()
    logging.shutdown()

if __name__ == "__main__":
    main()