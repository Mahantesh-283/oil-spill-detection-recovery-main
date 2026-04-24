import numpy as np

def latlon_to_dist(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two points 
    on the Earth using the Haversine formula.
    """
    # Earth's radius in kilometers
    R = 6371.0 
    
    # Convert degrees to radians
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    # Haversine calculation
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

def get_severity(volume):
    """Classifies the spill for the Response Report"""
    if volume > 10: return "CRITICAL"
    if volume > 2: return "HIGH"
    return "MODERATE"