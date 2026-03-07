def get_final_decision(risk_score):

    if risk_score < 0.30:
        return "LOW RISK 🟢"
    elif 0.30 < risk_score < 0.60:
        return "MEDIUM RISK 🟡 (monitor)"
    elif 0.60 < risk_score < 0.80:
        return "HIGH RISK 🟠 (Flag)"
    
    else:
        return "CRITICAL RISK 🔴 (likely fraud)"