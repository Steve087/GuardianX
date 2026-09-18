# ============================================================
# NightGuard AI - Incident Manager
# ============================================================

from uuid import uuid4


# ------------------------------------------------------------
# Create Incident
# ------------------------------------------------------------

def create_incident(event, risk_result):

    incident = {
        "incident_id": str(uuid4()),

        "event_type": event.get("event_type"),

        "track_id": event.get("track_id"),

        "timestamp": event.get("timestamp"),

        "confidence": event.get("confidence"),

        "zone": event.get("zone"),

        "direction": event.get("direction"),

        "risk_score": risk_result.get("risk_score"),

        "risk_level": risk_result.get("risk_level"),

        "reason": risk_result.get("reason")
    }

    return incident


# ------------------------------------------------------------
# Test
# ------------------------------------------------------------

if __name__ == "__main__":

    test_event = {
        "event_type": "line_crossing",
        "track_id": 7,
        "timestamp": "2026-09-18T10:00:00",
        "confidence": 0.82,
        "zone": None,
        "direction": "B -> A"
    }

    test_risk = {
        "risk_score": 59.6,
        "risk_level": "MEDIUM",
        "reason": "Person crossed Main Boundary (exit)"
    }

    incident = create_incident(
        test_event,
        test_risk
    )

    print("Incident:")
    print(incident)