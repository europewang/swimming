package com.example.swimming.logic

import com.example.swimming.model.CarCommand
import com.example.swimming.model.PersonDetection

class FollowLogic {
    companion object {
        private const val LEFT_THRESHOLD = 0.4f
        private const val RIGHT_THRESHOLD = 0.6f
        private const val STOP_AREA_THRESHOLD = 0.6f
    }

    fun determineCommand(detection: PersonDetection?): CarCommand {
        if (detection == null) {
            return CarCommand.STOP
        }

        val area = detection.width * detection.height
        if (area > STOP_AREA_THRESHOLD) {
            return CarCommand.STOP
        }

        return when {
            detection.centerX < LEFT_THRESHOLD -> CarCommand.LEFT
            detection.centerX > RIGHT_THRESHOLD -> CarCommand.RIGHT
            else -> CarCommand.FORWARD
        }
    }
}
