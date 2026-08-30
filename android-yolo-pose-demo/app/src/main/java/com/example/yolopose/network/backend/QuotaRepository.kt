package com.example.yolopose.network.backend

import android.content.Context

class QuotaRepository(
    private val authRepository: AuthRepository,
    context: Context,
    private val client: SimpleBackendClient = SimpleBackendClient(context)
) {
    fun getSummary(): Result<QuotaSummary> = runCatching {
        val response = client.get("/api/v1/quota/summary", authRepository.token())
        require(response.code == 0) { response.message.ifBlank { "quota summary failed" } }
        BackendJson.parseQuotaSummary(response.data ?: error("quota data missing"))
    }

    fun consume(sessionId: String, consumeType: String, consumeCount: Int): Result<ConsumeQuotaResponse> = runCatching {
        val response = client.post(
            path = "/api/v1/quota/consume",
            body = BackendJson.consumeQuotaRequestToJson(
                ConsumeQuotaRequest(
                    sessionId = sessionId,
                    consumeType = consumeType,
                    consumeCount = consumeCount
                )
            ),
            token = authRepository.token()
        )
        require(response.code == 0) { response.message.ifBlank { "quota consume failed" } }
        BackendJson.parseConsumeQuotaResponse(response.data ?: error("consume data missing"))
    }
}
