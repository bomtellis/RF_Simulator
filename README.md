# RF_Simulator
Ingests an IFC file of a building to predict wireless access point locations

## Features

- Loads an IFC file using IfcOpenShell.
- Extracts building storeys/floors and wall footprints.
- Lets you assign attenuation values by wall/material/type.
- Lets you place Wi-Fi access points on the floor plan.
- Simulates received signal strength using a log-distance path-loss model plus wall intersection losses.
- Displays a coloured RSSI heatmap over the selected floor.
- Exports the heatmap grid to CSV.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
python rf_simulator.py
```

## Notes

This is a 2D per-floor planning simulator. It uses wall intersections in plan to estimate attenuation. For a production version, add validated antenna patterns, AP height, slab attenuation, 5 GHz/6 GHz profiles, measured calibration, and better IFC geometry extraction for complex/curved elements.
