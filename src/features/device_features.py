def get_account_per_imei(device_sessions):
    return (
        device_sessions
        .groupby('IMEI')['CustomerID']
        .nunique()
        .rename('accounts_per_imei')
        .reset_index()
        
    )
    
def get_devices_per_customer(device_sessions):
    return (
        device_sessions
        .groupby('CustomerID')['IMEI']
        .nunique()
        .rename('devices_per_customer')
        .reset_index()
    )