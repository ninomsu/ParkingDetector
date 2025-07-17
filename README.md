# 🅿️ Edge-AI Parking Monitor

## 📖 Overview

This project implements an **Edge AI-based parking spot monitoring system** designed to run on a Raspberry Pi with a connected webcam. It detects when designated parking spots are **occupied or vacant** and sends real-time alerts to users.

The system is fully **observable** — exposing logs, metrics, and traces — which can be visualized using **Grafana**. Configuration is flexible, allowing users to tune detection parameters to fit their specific use case.

---

## ⚙️ Requirements

### 🧠 Edge Device (Raspberry Pi)
- Raspberry Pi 5
- USB webcam
- Python 3.x

### 💻 Host Machine (Monitoring)
- Docker + Docker Compose
- Web browser (for Grafana)

---

## 🛠️ Host Setup (Monitoring & Observability)

1. **Allow telemetry traffic**  
   Ensure ports `4317` and `4318` are open or exempted from firewall blocking.

2. **Start monitoring stack**  
   ```bash
   docker-compose up -d
   ```

3. **Access Grafana**  
   Navigate to [http://localhost:3000](http://localhost:3000)

4. **Set up Grafana**
   - Create an account (if prompted)
   - Add data sources:
     - **Prometheus**: `http://prometheus:9999`
     - **Loki**: `http://loki:3100`
     - **Tempo**: `http://tempo:3200`
   - Import the provided dashboard JSON  
     *(You may need to edit each panel to select the correct data source)*

---

## 📷 Edge Device Setup (Raspberry Pi)

1. **Create a Pushover account**  
   - Obtain an API key and user key for alerting.

2. **Create and activate a Python virtual environment**
   ```bash
   python3 -m venv ~/venv_parking
   source ~/venv_parking/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Tune Region of Interest (ROI)**
   - Run `camera_feed.py` to help determine correct ROI coordinates.
   - Update `config.xml` and `rois.xml` accordingly.
   - Keep these config files in the same directory as `main.py`.

5. **Run the main script**
   ```bash
   python main.py
   ```

---

## 📡 Log Forwarding with Fluent Bit

1. **Install Fluent Bit**  
   Follow setup instructions or build from source:  
   https://github.com/fluent/fluent-bit

2. **Configure Fluent Bit**  
   Edit `fluent-bit.conf` to match your telemetry setup.

3. **Run Fluent Bit**
   ```bash
   ./fluent-bit --config=fluent-bit.conf
   ```

---

## 📝 Notes

- Make sure the Raspberry Pi and host machine are on the same network.
- Observability is visible in Grafana, thanks to Loki, Prometheus, and Tempo integration.
