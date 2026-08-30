package com.example.yolopose.network.backend

import android.content.Context
import com.example.yolopose.network.BackendConfig
import com.example.yolopose.network.NetworkRouteSelector
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL
import org.json.JSONObject

class SimpleBackendClient(
    private val context: Context,
    private val baseUrl: String = BackendConfig.baseUrl
) {
    fun get(path: String, token: String? = null): ApiEnvelope<JSONObject> {
        val connection = openConnection(path, "GET", token)
        return execute(connection, null)
    }

    fun post(path: String, body: JSONObject, token: String? = null): ApiEnvelope<JSONObject> {
        val connection = openConnection(path, "POST", token)
        return execute(connection, body)
    }

    private fun openConnection(path: String, method: String, token: String?): HttpURLConnection {
        val normalizedBase = if (baseUrl.endsWith("/")) baseUrl.dropLast(1) else baseUrl
        val normalizedPath = if (path.startsWith("/")) path else "/$path"
        val url = URL("$normalizedBase$normalizedPath")
        val connection = (
            NetworkRouteSelector.findPreferredInternetNetwork(context)?.openConnection(url)
                ?: url.openConnection()
            ) as HttpURLConnection
        return connection.apply {
            requestMethod = method
            connectTimeout = BackendConfig.API_TIMEOUT_MS.toInt()
            readTimeout = BackendConfig.API_TIMEOUT_MS.toInt()
            doInput = true
            useCaches = false
            setRequestProperty("Content-Type", "application/json; charset=utf-8")
            setRequestProperty("Accept", "application/json")
            if (!token.isNullOrBlank()) {
                setRequestProperty("Authorization", "Bearer $token")
            }
            if (method != "GET") {
                doOutput = true
            }
        }
    }

    private fun execute(
        connection: HttpURLConnection,
        body: JSONObject?
    ): ApiEnvelope<JSONObject> {
        return try {
            if (body != null) {
                connection.outputStream.use { stream ->
                    stream.write(body.toString().toByteArray(Charsets.UTF_8))
                }
            }
            val statusCode = connection.responseCode
            val stream = if (statusCode in 200..299) {
                connection.inputStream
            } else {
                connection.errorStream ?: connection.inputStream
            }
            val payload = stream.bufferedReaderUtf8().use { it.readText() }
            val envelope = BackendJson.parseEnvelope(payload)
            if (statusCode !in 200..299 && envelope.code == 0) {
                envelope.copy(code = statusCode, message = "HTTP $statusCode")
            } else {
                envelope
            }
        } finally {
            connection.disconnect()
        }
    }

    private fun java.io.InputStream.bufferedReaderUtf8(): BufferedReader {
        return BufferedReader(InputStreamReader(this, Charsets.UTF_8))
    }
}
