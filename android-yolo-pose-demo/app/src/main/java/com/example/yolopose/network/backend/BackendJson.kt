package com.example.yolopose.network.backend

import org.json.JSONObject

object BackendJson {
    fun loginRequestToJson(request: LoginRequest): JSONObject = JSONObject().apply {
        put("loginType", request.loginType)
        put("account", request.account)
        put("password", request.password)
    }

    fun registerRequestToJson(request: RegisterRequest): JSONObject = JSONObject().apply {
        put("account", request.account)
        put("password", request.password)
        put("nickname", request.nickname)
    }

    fun deviceBindRequestToJson(request: DeviceBindRequest): JSONObject = JSONObject().apply {
        put("deviceCode", request.deviceCode)
        put("deviceName", request.deviceName)
        put("wifiSsid", request.wifiSsid)
        put("macAddress", request.macAddress)
    }

    fun sessionStartRequestToJson(request: SessionStartRequest): JSONObject = JSONObject().apply {
        put("sessionId", request.sessionId)
        put("deviceCode", request.deviceCode)
        put("mode", request.mode)
        put("videoSource", request.videoSource)
        put("appVersion", request.appVersion)
    }

    fun consumeQuotaRequestToJson(request: ConsumeQuotaRequest): JSONObject = JSONObject().apply {
        put("sessionId", request.sessionId)
        put("consumeType", request.consumeType)
        put("consumeCount", request.consumeCount)
    }

    fun sessionHeartbeatRequestToJson(request: SessionHeartbeatRequest): JSONObject = JSONObject().apply {
        put("sessionId", request.sessionId)
        put("trackStatus", request.trackStatus)
        put("fps", request.fps.toDouble())
        put("recording", request.recording)
        put("esp32Connected", request.esp32Connected)
    }

    fun sessionEndRequestToJson(request: SessionEndRequest): JSONObject = JSONObject().apply {
        put("sessionId", request.sessionId)
        put("durationSec", request.durationSec)
        put("avgFps", request.avgFps.toDouble())
        put("recordFilePath", request.recordFilePath)
        put("resultSummary", request.resultSummary)
    }

    fun parseEnvelope(raw: String): ApiEnvelope<JSONObject> {
        val root = JSONObject(raw)
        return ApiEnvelope(
            code = root.optInt("code", -1),
            message = root.optString("message", ""),
            data = root.optJSONObject("data")
        )
    }

    fun parseLoginResponse(data: JSONObject): LoginResponse = LoginResponse(
        userId = data.optLong("userId", 0L),
        token = data.optString("token", ""),
        nickname = data.optString("nickname", ""),
        quotaRemain = data.optInt("quotaRemain", 0),
        planType = data.optString("planType", "")
    )

    fun parseUserProfile(data: JSONObject): UserProfile = UserProfile(
        userId = data.optLong("userId", 0L),
        nickname = data.optString("nickname", ""),
        avatar = data.optString("avatar", "").ifBlank { null },
        quotaRemain = data.optInt("quotaRemain", 0),
        quotaTotal = data.optInt("quotaTotal", 0),
        planType = data.optString("planType", ""),
        planExpireAt = data.optString("planExpireAt", "").ifBlank { null }
    )

    fun parseQuotaSummary(data: JSONObject): QuotaSummary = QuotaSummary(
        quotaTotal = data.optInt("quotaTotal", 0),
        quotaUsed = data.optInt("quotaUsed", 0),
        quotaRemain = data.optInt("quotaRemain", 0),
        resetRule = data.optString("resetRule", "").ifBlank { null }
    )

    fun parseSessionStartResponse(data: JSONObject): SessionStartResponse = SessionStartResponse(
        sessionId = data.optString("sessionId", ""),
        accepted = data.optBoolean("accepted", true)
    )

    fun parseConsumeQuotaResponse(data: JSONObject): ConsumeQuotaResponse = ConsumeQuotaResponse(
        success = data.optBoolean("success", true),
        quotaRemain = data.optInt("quotaRemain", 0)
    )

    fun parseDeviceRuntimeStatus(data: JSONObject): DeviceRuntimeStatus = DeviceRuntimeStatus(
        deviceCode = data.optString("deviceCode", data.optString("device_code", "")),
        currentIp = data.optString("currentIp", data.optString("current_ip", "")).ifBlank { null },
        currentHttpPort = data.optInt("currentHttpPort", data.optInt("current_http_port", 80)),
        currentStreamPort = data.optInt("currentStreamPort", data.optInt("current_stream_port", 81)),
        onlineStatus = data.optString("onlineStatus", data.optString("online_status", "offline")),
        ssid = data.optString("ssid", "").ifBlank { null },
        lastSeenAt = data.optString("lastSeenAt", data.optString("last_seen_at", "")).ifBlank { null }
    )
}
