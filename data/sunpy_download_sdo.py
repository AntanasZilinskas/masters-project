from sunpy.net import Fido, attrs as a

# Narrow the search to a short time window to limit the total number of files
result = Fido.search(
    a.Time("2022-01-01 00:00", "2022-01-01 01:00"),  # 1-hour window
    a.Instrument.hmi,
    a.Physobs.los_magnetic_field
)

# Download the data found in this narrowed time range
files = Fido.fetch(result)
print(f"Downloaded files (narrowed timeframe): {files}")
