document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadOverlay = document.getElementById('upload-overlay');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resetBtn = document.getElementById('reset-btn');

    // Initialize Map with Real World Coordinates (OpenStreetMap)
    const map = L.map('map').setView([33.5731, -7.5898], 13); // Default to Casablanca approx

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
        subdomains: 'abcd',
        maxZoom: 20
    }).addTo(map);

    let currentGeoJsonLayer = null;

    // --- Drag & Drop Logic ---
    dropZone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    });

    fileInput.addEventListener('change', function() {
        if (this.files[0]) handleFile(this.files[0]);
    });

    resetBtn.addEventListener('click', () => {
        uploadOverlay.classList.add('active');
        resetBtn.classList.add('hidden');
        if (currentGeoJsonLayer) map.removeLayer(currentGeoJsonLayer);
    });

    // --- Core Processing ---
    async function handleFile(file) {
        // UI State: Loading
        uploadOverlay.classList.remove('active');
        loadingOverlay.classList.add('active');

        // Call API
        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Server error');
            }

            // API Success - Plot on Map
            plotResults(data.geojson);

        } catch (error) {
            alert("AI Processing Failed: " + error.message);
            uploadOverlay.classList.add('active');
        } finally {
            loadingOverlay.classList.remove('active');
        }
    }

    function plotResults(geojsonData) {
        // Clear previous
        if (currentGeoJsonLayer) map.removeLayer(currentGeoJsonLayer);

        // Add GeoJSON
        const geojsonStyle = {
            "color": "#ef4444",      // Slum Red
            "weight": 2,
            "opacity": 0.8,
            "fillColor": "#ef4444",
            "fillOpacity": 0.4
        };

        // Note: The native EPSG is 3857, which requires projection handling in Leaflet,
        // but let's test if Leaflet handles it out of the box or if we need proj4js.
        // Actually, Leaflet geoJSON expects EPSG:4326 (Lat/Lon).
        // Let's just load it. If it fails, we will update the python script to reproject to EPSG:4326.
        currentGeoJsonLayer = L.geoJSON(geojsonData, {
            style: geojsonStyle,
            onEachFeature: function(feature, layer) {
                layer.bindPopup("<b>AI Detection</b><br>Informal Settlement (Slum)");
                layer.on({
                    mouseover: function(e) {
                        const layer = e.target;
                        layer.setStyle({ fillOpacity: 0.7, weight: 3 });
                    },
                    mouseout: function(e) {
                        currentGeoJsonLayer.resetStyle(e.target);
                    }
                });
            }
        }).addTo(map);
        
        try {
            map.fitBounds(currentGeoJsonLayer.getBounds());
        } catch(e) {
            console.log("Could not fit bounds, possibly due to CRS mismatch.");
        }

        // UI State: Success
        resetBtn.classList.remove('hidden');
    }
});
