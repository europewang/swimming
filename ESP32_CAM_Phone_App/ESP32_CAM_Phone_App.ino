#include <WiFi.h>
#include <Preferences.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include "esp_camera.h"
#include "board_config.h"

// Functions implemented in app_httpd.cpp
void startCameraServer();
void stopCameraServer();
void setupLedFlash();

static bool cameraServerStarted = false;
static bool bleProvisioningActive = false;
static bool pendingCameraStartup = false;

static Preferences preferences;
static BLECharacteristic *provisionNotifyCharacteristic = nullptr;
static bool bleClientConnected = false;
static bool stationConnectInProgress = false;
static String stationSsid;
static String stationPassword;
static unsigned long lastStationAttemptAt = 0;
static unsigned long lastWifiConnectedAt = 0;
static String pendingReadyIp;
static int pendingReadyNotifyCount = 0;
static unsigned long lastReadyNotifyAt = 0;
static unsigned long lastBackendReportAt = 0;
static unsigned long lastBackendReportAttemptAt = 0;
static unsigned long lastDisconnectEventAt = 0;
static uint8_t lastDisconnectReason = 0;
static bool backendReportPending = false;
static String pendingBackendStatus = "offline";
static String lastReportedIp = "";
static String lastReportedStatus = "";

static const char *PREF_NAMESPACE = "swimcam";
static const char *PREF_KEY_SSID = "sta_ssid";
static const char *PREF_KEY_PASSWORD = "sta_pwd";
static const unsigned long STATION_RETRY_INTERVAL_MS = 12000UL;
static const unsigned long BLE_STOP_TO_CAMERA_DELAY_MS = 1200UL;
static const unsigned long BLE_STOP_MAX_WAIT_MS = 4000UL;
static const unsigned long READY_NOTIFY_INTERVAL_MS = 250UL;
static const int READY_NOTIFY_REPEAT_COUNT = 4;
static const unsigned long BACKEND_REPORT_INTERVAL_MS = 10000UL;
static const unsigned long BACKEND_REPORT_RETRY_INTERVAL_MS = 2000UL;
static const char *DEVICE_CODE = "esp32cam-001";
static const char *DEVICE_NAME = "Pool Cam 1";
static const char *BACKEND_RUNTIME_REPORT_URL = "http://43.156.49.149:8085/api/v1/device/runtime/report";
static const char *BACKEND_HOST = "43.156.49.149";
static const uint16_t BACKEND_PORT = 8085;
static const char *BACKEND_RUNTIME_REPORT_PATH = "/api/v1/device/runtime/report";
static BLEUUID SERVICE_UUID("0000FF10-0000-1000-8000-00805F9B34FB");
static BLEUUID PROVISION_WRITE_UUID("0000FF11-0000-1000-8000-00805F9B34FB");
static BLEUUID PROVISION_NOTIFY_UUID("0000FF12-0000-1000-8000-00805F9B34FB");

enum DeviceState {
  STATE_BOOT = 0,
  STATE_BLE_ONLY,
  STATE_WIFI_STA_CONNECT,
  STATE_STOP_BLE,
  STATE_CAMERA_ON,
  STATE_RUNNING
};

static DeviceState currentState = STATE_BOOT;
static unsigned long stateEnteredAtMs = 0;

static void notifyProvisionState(const String &payload);
static void connectToConfiguredHotspot(bool forceReconnect);
static void loadProvisionedCredentials();
static void saveProvisionedCredentials(const String &ssid, const String &password);
static bool handleProvisioningPayload(const String &payload);
static String extractJsonString(const String &payload, const char *key);
static void handleWiFiEvent(WiFiEvent_t event, WiFiEventInfo_t info);
static void startCameraSubsystemIfNeeded();
static void stopCameraSubsystemIfNeeded();
static void stopBleProvisioning();
static void returnToBleProvisioningMode(const char *reason);
static void scheduleReadyIpBroadcast(const String &ipString);
static void flushReadyIpBroadcastIfNeeded();
static bool reportRuntimeStatusToBackend(const String &onlineStatus);
static void reportRuntimeStatusIfNeeded();
static void scheduleBackendRuntimeReport(const String &status);
static void enterState(DeviceState newState, const char *reason);
static const char *stateName(DeviceState state);
static void logState(const String &message);
static String escapeJsonString(const String &value);

static const char *stateName(DeviceState state) {
  switch (state) {
    case STATE_BOOT:
      return "BOOT";
    case STATE_BLE_ONLY:
      return "BLE_ONLY";
    case STATE_WIFI_STA_CONNECT:
      return "WIFI_STA";
    case STATE_STOP_BLE:
      return "STOP_BLE";
    case STATE_CAMERA_ON:
      return "CAMERA_ON";
    case STATE_RUNNING:
      return "RUNNING";
    default:
      return "UNKNOWN";
  }
}

static const char *disconnectReasonToName(uint8_t reason) {
  return WiFi.disconnectReasonName((wifi_err_reason_t)reason);
}

static void logState(const String &message) {
  Serial.printf("[STATE:%s][%lu ms] %s\n", stateName(currentState), millis(), message.c_str());
}

static void enterState(DeviceState newState, const char *reason) {
  currentState = newState;
  stateEnteredAtMs = millis();
  Serial.printf("\n>>> ENTER STATE: %s | reason=%s | t=%lu ms\n",
                stateName(newState),
                reason != nullptr ? reason : "none",
                stateEnteredAtMs);
}

class ProvisionServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *server) override {
    bleClientConnected = true;
    logState("BLE client connected");
    if (server != nullptr) {
      server->getAdvertising()->stop();
    }
  }

  void onDisconnect(BLEServer *server) override {
    bleClientConnected = false;
    logState("BLE client disconnected");
    if (server != nullptr) {
      server->getAdvertising()->start();
    }
  }
};

class ProvisionWriteCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *characteristic) override {
    String payload = characteristic->getValue();
    if (payload.isEmpty()) {
      return;
    }
    logState(String("BLE provisioning payload: ") + payload);
    if (!handleProvisioningPayload(payload)) {
      notifyProvisionState("{\"status\":\"error\",\"message\":\"invalid_payload\"}");
    }
  }
};

static void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Good starting point for real-time phone-side CV:
  // QVGA (320x240) is low-latency and light enough for ESP32-CAM.
  if (psramFound()) {
    config.frame_size = FRAMESIZE_VGA;      // 640x480; App may lower via /control
    config.jpeg_quality = 12;               // lower number = better JPEG quality
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;  // reduce latency for live processing
    config.fb_location = CAMERA_FB_IN_PSRAM;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 14;
    config.fb_count = 1;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    config.fb_location = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    delay(3000);
    ESP.restart();
  }

  sensor_t *s = esp_camera_sensor_get();

  // Default to QVGA for lower latency. You can change this from the App later.
  if (s != nullptr && s->pixformat == PIXFORMAT_JPEG) {
    s->set_framesize(s, FRAMESIZE_QVGA);    // 320x240 for CV / target tracking
    s->set_quality(s, 12);
  }

#if defined(LED_GPIO_NUM)
  setupLedFlash();
#endif
  logState("Camera initialized");
}

static void loadProvisionedCredentials() {
  preferences.begin(PREF_NAMESPACE, true);
  stationSsid = preferences.getString(PREF_KEY_SSID, "");
  stationPassword = preferences.getString(PREF_KEY_PASSWORD, "");
  preferences.end();
  if (stationSsid.isEmpty()) {
    logState("No stored hotspot credentials found in Preferences");
  } else {
    logState(String("Loaded stored hotspot SSID: ") + stationSsid);
  }
}

static void saveProvisionedCredentials(const String &ssid, const String &password) {
  preferences.begin(PREF_NAMESPACE, false);
  preferences.putString(PREF_KEY_SSID, ssid);
  preferences.putString(PREF_KEY_PASSWORD, password);
  preferences.end();
  stationSsid = ssid;
  stationPassword = password;
  logState(String("Saved hotspot credentials for SSID: ") + ssid);
}

static void notifyProvisionState(const String &payload) {
  logState(String("Provision notify: ") + payload);
  if (provisionNotifyCharacteristic != nullptr && bleClientConnected) {
    provisionNotifyCharacteristic->setValue(payload.c_str());
    provisionNotifyCharacteristic->notify();
  }
}

static String extractJsonString(const String &payload, const char *key) {
  String pattern = String("\"") + key + "\":";
  int keyIndex = payload.indexOf(pattern);
  if (keyIndex < 0) {
    return "";
  }
  int valueStart = payload.indexOf('"', keyIndex + pattern.length());
  if (valueStart < 0) {
    return "";
  }
  int valueEnd = payload.indexOf('"', valueStart + 1);
  if (valueEnd < 0) {
    return "";
  }
  return payload.substring(valueStart + 1, valueEnd);
}

static bool handleProvisioningPayload(const String &payload) {
  String command = extractJsonString(payload, "cmd");
  if (command != "provision_wifi") {
    return false;
  }
  String ssid = extractJsonString(payload, "ssid");
  String password = extractJsonString(payload, "password");
  if (ssid.isEmpty()) {
    notifyProvisionState("{\"status\":\"error\",\"message\":\"missing_ssid\"}");
    return true;
  }

  saveProvisionedCredentials(ssid, password);
  notifyProvisionState(String("{\"status\":\"saved\",\"ssid\":\"") + escapeJsonString(ssid) + "\"}");
  connectToConfiguredHotspot(true);
  return true;
}

static void connectToConfiguredHotspot(bool forceReconnect) {
  if (stationSsid.isEmpty()) {
    logState("No stored hotspot credentials, skip STA connection");
    return;
  }
  if (!forceReconnect && WiFi.status() == WL_CONNECTED) {
    logState("STA already connected, skip reconnect");
    return;
  }

  enterState(STATE_WIFI_STA_CONNECT, forceReconnect ? "received_new_credentials" : "connect_or_retry");
  stationConnectInProgress = true;
  lastStationAttemptAt = millis();
  logState(String("Connecting to phone hotspot: ") + stationSsid);
  notifyProvisionState(String("{\"status\":\"connecting\",\"ssid\":\"") + escapeJsonString(stationSsid) + "\"}");

  WiFi.disconnect(true, false);
  delay(100);
  WiFi.begin(stationSsid.c_str(), stationPassword.c_str());
}

static void handleWiFiEvent(WiFiEvent_t event, WiFiEventInfo_t info) {
  switch (event) {
    case ARDUINO_EVENT_WIFI_STA_CONNECTED:
      logState(String("STA connected to hotspot: ") + stationSsid);
      notifyProvisionState(String("{\"status\":\"sta_connected\",\"ssid\":\"") + escapeJsonString(stationSsid) + "\"}");
      break;
    case ARDUINO_EVENT_WIFI_STA_GOT_IP: {
      stationConnectInProgress = false;
      lastWifiConnectedAt = millis();
      IPAddress ip = WiFi.localIP();
      String ipString = ip.toString();
      logState(String("STA got IP: ") + ipString);
      notifyProvisionState(
        String("{\"status\":\"ready\",\"ip\":\"") + escapeJsonString(ipString) +
        "\",\"host\":\"" + escapeJsonString(ipString) + "\"}"
      );
      scheduleReadyIpBroadcast(ipString);
      scheduleBackendRuntimeReport("online");
      pendingCameraStartup = true;
      enterState(STATE_STOP_BLE, "wifi_connected_ready_to_stop_ble");
      break;
    }
    case ARDUINO_EVENT_WIFI_STA_DISCONNECTED: {
      uint8_t reason = info.wifi_sta_disconnected.reason;
      if (!reason) {
        reason = WIFI_REASON_UNSPECIFIED;
      }
      const unsigned long now = millis();
      if (lastDisconnectReason == reason && lastDisconnectEventAt != 0 && now - lastDisconnectEventAt < 1200UL) {
        logState(String("Ignore duplicate STA disconnect event, reason=") + reason + " (" + disconnectReasonToName(reason) + ")");
        return;
      }
      lastDisconnectReason = reason;
      lastDisconnectEventAt = now;
      if (stationConnectInProgress || !stationSsid.isEmpty()) {
        stationConnectInProgress = false;
        logState(
          String("STA disconnected from hotspot, reason=") + reason +
          " (" + disconnectReasonToName(reason) + ")"
        );
        notifyProvisionState(
          String("{\"status\":\"error\",\"message\":\"sta_disconnected\",\"ssid\":\"") + escapeJsonString(stationSsid) +
          "\",\"reason\":" + reason +
          ",\"reason_name\":\"" + escapeJsonString(disconnectReasonToName(reason)) + "\"}"
        );
        if (cameraServerStarted) {
          returnToBleProvisioningMode("wifi_disconnected_after_running");
        } else {
          enterState(STATE_BLE_ONLY, "wifi_disconnected_before_camera");
        }
      }
      break;
    }
    default:
      break;
  }
}

static void startBleProvisioning() {
  if (bleProvisioningActive) {
    logState("BLE provisioning already active");
    return;
  }
  BLEDevice::init("ESP32-CAM-PROV");
  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new ProvisionServerCallbacks());

  BLEService *service = server->createService(SERVICE_UUID);
  BLECharacteristic *writeCharacteristic = service->createCharacteristic(
    PROVISION_WRITE_UUID,
    BLECharacteristic::PROPERTY_WRITE
  );
  provisionNotifyCharacteristic = service->createCharacteristic(
    PROVISION_NOTIFY_UUID,
    BLECharacteristic::PROPERTY_NOTIFY | BLECharacteristic::PROPERTY_READ
  );
  provisionNotifyCharacteristic->addDescriptor(new BLE2902());
  writeCharacteristic->setCallbacks(new ProvisionWriteCallbacks());

  service->start();
  server->getAdvertising()->addServiceUUID(SERVICE_UUID);
  server->getAdvertising()->start();
  bleProvisioningActive = true;
  enterState(STATE_BLE_ONLY, "ble_service_started");
  logState("BLE provisioning service started");
}

static void stopBleProvisioning() {
  if (!bleProvisioningActive) {
    return;
  }
  delay(600);
  BLEDevice::deinit(false);
  bleProvisioningActive = false;
  provisionNotifyCharacteristic = nullptr;
  bleClientConnected = false;
  pendingReadyIp = "";
  pendingReadyNotifyCount = 0;
  logState("BLE provisioning service stopped");
}

static void startCameraSubsystemIfNeeded() {
  if (cameraServerStarted) {
    return;
  }
  enterState(STATE_CAMERA_ON, "starting_camera_subsystem");
  initCamera();
  startCameraServer();
  cameraServerStarted = true;
  IPAddress ip = WiFi.localIP();
  String ipString = ip.toString();
  Serial.println("========================================");
  Serial.printf("Camera server IP : %s\n", ipString.c_str());
  Serial.printf("Snapshot        : http://%s/capture\n", ipString.c_str());
  Serial.printf("Stream          : http://%s:81/stream\n", ipString.c_str());
  Serial.println("========================================");
  logState("Camera server started");
  enterState(STATE_RUNNING, "camera_server_running");
}

static void stopCameraSubsystemIfNeeded() {
  if (!cameraServerStarted) {
    return;
  }
  logState("Stopping camera server");
  stopCameraServer();
  esp_camera_deinit();
  cameraServerStarted = false;
  pendingCameraStartup = false;
  logState("Camera subsystem stopped");
}

static void returnToBleProvisioningMode(const char *reason) {
  logState(String("Returning to BLE provisioning mode, reason=") + (reason != nullptr ? reason : "unknown"));
  stopCameraSubsystemIfNeeded();
  if (WiFi.status() == WL_CONNECTED) {
    reportRuntimeStatusToBackend("offline");
  }
  WiFi.disconnect(true, false);
  delay(100);
  WiFi.mode(WIFI_OFF);
  delay(150);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  stationConnectInProgress = false;
  lastStationAttemptAt = millis();
  pendingReadyIp = "";
  pendingReadyNotifyCount = 0;
  if (!bleProvisioningActive) {
    startBleProvisioning();
  } else {
    enterState(STATE_BLE_ONLY, reason);
  }
}

static void scheduleReadyIpBroadcast(const String &ipString) {
  pendingReadyIp = ipString;
  pendingReadyNotifyCount = 0;
  lastReadyNotifyAt = 0;
  logState(String("Schedule repeated ready notify for IP: ") + ipString);
}

static void flushReadyIpBroadcastIfNeeded() {
  if (!bleProvisioningActive || !bleClientConnected || pendingReadyIp.isEmpty()) {
    return;
  }
  if (pendingReadyNotifyCount >= READY_NOTIFY_REPEAT_COUNT) {
    return;
  }
  const unsigned long now = millis();
  if (lastReadyNotifyAt != 0 && now - lastReadyNotifyAt < READY_NOTIFY_INTERVAL_MS) {
    return;
  }
  lastReadyNotifyAt = now;
  pendingReadyNotifyCount += 1;
  notifyProvisionState(
    String("{\"status\":\"ready\",\"ip\":\"") + escapeJsonString(pendingReadyIp) +
    "\",\"host\":\"" + escapeJsonString(pendingReadyIp) +
    "\",\"seq\":" + pendingReadyNotifyCount + "}"
  );
  logState(String("Repeated ready notify seq=") + pendingReadyNotifyCount);
}

static void scheduleBackendRuntimeReport(const String &status) {
  pendingBackendStatus = status;
  backendReportPending = true;
  lastBackendReportAttemptAt = 0;
  logState(String("Schedule backend runtime report, status=") + status);
}

static String escapeJsonString(const String &value) {
  String escaped;
  escaped.reserve(value.length() + 8);
  for (size_t i = 0; i < value.length(); ++i) {
    char c = value.charAt(i);
    switch (c) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += c;
        break;
    }
  }
  return escaped;
}

static bool reportRuntimeStatusToBackend(const String &onlineStatus) {
  if (WiFi.status() != WL_CONNECTED && onlineStatus == "online") {
    return false;
  }
  String ipString = WiFi.status() == WL_CONNECTED ? WiFi.localIP().toString() : "";
  String body = String("{") +
                "\"device_code\":\"" + escapeJsonString(DEVICE_CODE) + "\"," +
                "\"device_name\":\"" + escapeJsonString(DEVICE_NAME) + "\"," +
                "\"current_ip\":\"" + escapeJsonString(ipString) + "\"," +
                "\"current_http_port\":80," +
                "\"current_stream_port\":81," +
                "\"online_status\":\"" + escapeJsonString(onlineStatus) + "\"," +
                "\"network_type\":\"phone_hotspot\"," +
                "\"ssid\":\"" + escapeJsonString(stationSsid) + "\"" +
                "}";
  WiFiClient client;
  client.setTimeout(3500);
  if (!client.connect(BACKEND_HOST, BACKEND_PORT)) {
    logState(String("Backend runtime report connect failed: ") + BACKEND_HOST + ":" + BACKEND_PORT);
    lastBackendReportAttemptAt = millis();
    return false;
  }

  client.print(String("POST ") + BACKEND_RUNTIME_REPORT_PATH + " HTTP/1.1\r\n");
  client.print(String("Host: ") + BACKEND_HOST + ":" + BACKEND_PORT + "\r\n");
  client.print("Content-Type: application/json\r\n");
  client.print("Connection: close\r\n");
  client.print(String("Content-Length: ") + body.length() + "\r\n\r\n");
  client.print(body);

  unsigned long startedAt = millis();
  while (client.connected() && !client.available() && millis() - startedAt < 3500UL) {
    delay(10);
  }

  String response;
  String firstLine;
  bool firstLineCaptured = false;
  while (client.available()) {
    String line = client.readStringUntil('\n');
    if (!firstLineCaptured && line.length() > 0) {
      firstLine = line;
      firstLineCaptured = true;
    }
    response += line;
  }
  client.stop();
  const bool ok = firstLine.indexOf("200") >= 0;
  logState(
    String("Backend runtime report to ") + BACKEND_RUNTIME_REPORT_URL +
    ", ok=" + (ok ? "true" : "false") +
    ", firstLine=" + firstLine +
    ", response=" + response
  );
  lastBackendReportAttemptAt = millis();
  if (!ok) {
    return false;
  }
  lastBackendReportAt = millis();
  backendReportPending = false;
  lastReportedIp = ipString;
  lastReportedStatus = onlineStatus;
  return true;
}

static void reportRuntimeStatusIfNeeded() {
  const String currentIp = WiFi.status() == WL_CONNECTED ? WiFi.localIP().toString() : "";
  if (backendReportPending) {
    if (pendingBackendStatus == "online" && WiFi.status() != WL_CONNECTED) {
      return;
    }
    if (lastBackendReportAttemptAt != 0 &&
        millis() - lastBackendReportAttemptAt < BACKEND_REPORT_RETRY_INTERVAL_MS) {
      return;
    }
    reportRuntimeStatusToBackend(pendingBackendStatus);
    return;
  }

  if (WiFi.status() != WL_CONNECTED) {
    return;
  }

  if (currentIp != lastReportedIp || lastReportedStatus != "online") {
    scheduleBackendRuntimeReport("online");
    return;
  }

  if (lastBackendReportAt != 0 && millis() - lastBackendReportAt < BACKEND_REPORT_INTERVAL_MS) {
    return;
  }
  scheduleBackendRuntimeReport("online");
}

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  delay(500);
  enterState(STATE_BOOT, "setup_begin");

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.onEvent(handleWiFiEvent);
  logState("WiFi set to STA mode only");
  loadProvisionedCredentials();
  startBleProvisioning();
  connectToConfiguredHotspot(false);
}

void loop() {
  flushReadyIpBroadcastIfNeeded();
  reportRuntimeStatusIfNeeded();
  if (pendingCameraStartup && WiFi.status() == WL_CONNECTED && !cameraServerStarted) {
    if (bleProvisioningActive) {
      const bool readyNotifyFinished = pendingReadyNotifyCount >= READY_NOTIFY_REPEAT_COUNT;
      const bool noBleClientToWaitFor = !bleClientConnected;
      const bool stopBleWaitTimedOut = millis() - stateEnteredAtMs >= BLE_STOP_MAX_WAIT_MS;
      if ((readyNotifyFinished && millis() - stateEnteredAtMs >= BLE_STOP_TO_CAMERA_DELAY_MS) ||
          (noBleClientToWaitFor && millis() - stateEnteredAtMs >= BLE_STOP_TO_CAMERA_DELAY_MS) ||
          stopBleWaitTimedOut) {
        logState(
          String("Proceed to stop BLE and start camera, readyNotifyFinished=") + (readyNotifyFinished ? "true" : "false") +
          ", bleClientConnected=" + (bleClientConnected ? String("true") : String("false")) +
          ", pendingReadyNotifyCount=" + pendingReadyNotifyCount
        );
        pendingCameraStartup = false;
        stopBleProvisioning();
        startCameraSubsystemIfNeeded();
      }
    } else {
      pendingCameraStartup = false;
      startCameraSubsystemIfNeeded();
    }
  }
  if (!stationSsid.isEmpty() &&
      WiFi.status() != WL_CONNECTED &&
      !stationConnectInProgress &&
      !cameraServerStarted &&
      millis() - lastStationAttemptAt > STATION_RETRY_INTERVAL_MS) {
    logState("STA retry timer reached before camera startup, trying WiFi again");
    connectToConfiguredHotspot(false);
  }
  delay(1000);
}
