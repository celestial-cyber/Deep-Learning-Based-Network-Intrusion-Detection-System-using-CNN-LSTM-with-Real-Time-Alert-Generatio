from datetime import datetime


# -----------------------------
# SEVERITY ENGINE (IMPROVED)
# -----------------------------
def get_severity(confidence, prediction):
    """
    Real IDS-style severity logic:
    combines prediction + confidence
    """

    if prediction == 0:
        return "LOW"

    if confidence >= 0.95:
        return "CRITICAL"
    elif confidence >= 0.85:
        return "HIGH"
    elif confidence >= 0.70:
        return "MEDIUM"
    else:
        return "LOW"


# -----------------------------
# ATTACK TYPE MAPPING (FIXED - NO RANDOMNESS)
# -----------------------------
def get_attack_type(prediction):
    """
    Deterministic mapping (REAL SYSTEM STYLE)
    Ready for multi-class upgrade
    """

    attack_map = {
        0: "Normal Traffic",
        1: "IoT Botnet / Traffic Anomaly"
    }

    return attack_map.get(prediction, "Unknown Traffic")


# -----------------------------
# ACTION ENGINE (REAL IDS STYLE)
# -----------------------------
def get_action(prediction, confidence):

    if prediction == 0:
        return "ALLOW TRAFFIC"

    if confidence >= 0.95:
        return "IMMEDIATE BLOCK + ALERT ADMIN"
    elif confidence >= 0.85:
        return "BLOCK IP + LOG INCIDENT"
    elif confidence >= 0.70:
        return "QUARANTINE DEVICE / MONITOR"
    else:
        return "MONITOR ONLY"


# -----------------------------
# MAIN ALERT GENERATOR
# -----------------------------
def generate_alert(prediction, confidence, source_ip="Unknown", target="IoT Gateway"):

    status = "ATTACK DETECTED" if prediction == 1 else "NORMAL TRAFFIC"

    alert = {
        "status": status,
        "type": get_attack_type(prediction),
        "severity": get_severity(confidence, prediction),
        "confidence": round(float(confidence), 4),
        "source_ip": source_ip,
        "target": target,
        "action": get_action(prediction, confidence),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return alert