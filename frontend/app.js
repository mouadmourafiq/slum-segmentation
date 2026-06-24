document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadSection = document.querySelector('.upload-section');
    const dashboardStats = document.getElementById('dashboard-stats');
    const layerControls = document.getElementById('layer-controls');
    const scanningOverlay = document.getElementById('scanning-overlay');
    const resetBtn = document.getElementById('reset-btn');
    
    // Stats Elements
    const statModel = document.getElementById('stat-model');
    const statTime = document.getElementById('stat-time');
    const statPolygons = document.getElementById('stat-polygons');
    
    // Toggle
    const togglePolygons = document.getElementById('toggle-polygons');

    // Report elements
    const actionButtons = document.getElementById('action-buttons');
    const reportBtn = document.getElementById('report-btn');
    const reportModal = document.getElementById('report-modal');
    const closeModalBtn = document.getElementById('close-modal');
    const reportTableBody = document.querySelector('#report-table tbody');
    const exportCsvBtn = document.getElementById('export-csv-btn');

    // Initialize Map with Real World Coordinates (OpenStreetMap)
    const map = L.map('map', {zoomControl: false}).setView([0, 0], 2); // Start zoomed out
    L.control.zoom({ position: 'bottomright' }).addTo(map);

    // Basemap toggle
    const basemapToggleBtn = document.getElementById('basemap-toggle');
    let isSatellite = false;

    const darkLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
        subdomains: 'abcd',
        maxZoom: 20
    });
    
    const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri',
        maxZoom: 20
    });

    darkLayer.addTo(map); // default

    basemapToggleBtn.addEventListener('click', () => {
        if (isSatellite) {
            map.removeLayer(satelliteLayer);
            darkLayer.addTo(map);
            basemapToggleBtn.innerHTML = '🌍 Satellite View';
            isSatellite = false;
        } else {
            map.removeLayer(darkLayer);
            satelliteLayer.addTo(map);
            basemapToggleBtn.innerHTML = '🌑 Dark Mode';
            isSatellite = true;
        }
    });

    let currentGeoJsonLayer = null;
    let boundingBoxLayer = null;
    let reportData = [];

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
        uploadSection.classList.remove('hidden');
        dashboardStats.classList.add('hidden');
        layerControls.classList.add('hidden');
        actionButtons.classList.add('hidden');
        
        if (currentGeoJsonLayer) map.removeLayer(currentGeoJsonLayer);
        if (boundingBoxLayer) map.removeLayer(boundingBoxLayer);
        map.setView([0, 0], 2); // Zoom out
    });

    togglePolygons.addEventListener('change', (e) => {
        if (!currentGeoJsonLayer) return;
        if (e.target.checked) {
            map.addLayer(currentGeoJsonLayer);
            if(boundingBoxLayer) map.addLayer(boundingBoxLayer);
        } else {
            map.removeLayer(currentGeoJsonLayer);
            if(boundingBoxLayer) map.removeLayer(boundingBoxLayer);
        }
    });

    // --- Report Logic ---
    reportBtn.addEventListener('click', () => {
        reportModal.classList.remove('hidden');
        populateTable();
    });

    closeModalBtn.addEventListener('click', () => {
        reportModal.classList.add('hidden');
    });

    exportCsvBtn.addEventListener('click', () => {
        if (reportData.length === 0) return;
        let csvContent = "data:text/csv;charset=utf-8,ID,Area(m2),Latitude,Longitude\n";
        reportData.forEach(row => {
            csvContent += `${row.id},${row.area},${row.lat},${row.lng}\n`;
        });
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "slum_coordinates_report.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    function populateTable() {
        reportTableBody.innerHTML = '';
        reportData.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${row.id}</strong></td>
                <td>${row.area}</td>
                <td>${row.lat}</td>
                <td>${row.lng}</td>
                <td><button class="btn btn-secondary" style="padding: 0.5rem;" onclick="map.setView([${row.lat}, ${row.lng}], 18); document.getElementById('close-modal').click();">Zoom In 🔎</button></td>
            `;
            reportTableBody.appendChild(tr);
        });
    }

    // --- Core Processing ---
    async function handleFile(file) {
        // UI State: Scanning Animation
        uploadSection.classList.add('hidden');
        scanningOverlay.classList.remove('hidden');

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

            // Update Stats UI
            const meta = data.metadata || {};
            statModel.innerText = meta.model_name || "Region-Aware MoE";
            statTime.innerText = meta.inference_time_seconds ? `${meta.inference_time_seconds}s` : "Fast";
            statPolygons.innerText = meta.polygon_count || "0";

            // API Success - Plot on Map
            plotResults(data.geojson);

        } catch (error) {
            alert("AI Processing Failed: " + error.message);
            uploadSection.classList.remove('hidden');
        } finally {
            scanningOverlay.classList.add('hidden');
        }
    }

    function plotResults(geojsonData) {
        // Clear previous
        if (currentGeoJsonLayer) map.removeLayer(currentGeoJsonLayer);
        if (boundingBoxLayer) map.removeLayer(boundingBoxLayer);
        reportData = [];

        // Add GeoJSON
        const geojsonStyle = {
            "color": "#ef4444",      // Slum Red
            "weight": 2,
            "opacity": 0.8,
            "fillColor": "#ef4444",
            "fillOpacity": 0.4
        };

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
            const bounds = currentGeoJsonLayer.getBounds();
            // Draw Bounding Box
            boundingBoxLayer = L.rectangle(bounds, {
                color: "#facc15", 
                weight: 3, 
                fillOpacity: 0, 
                dashArray: "10, 10"
            }).addTo(map);

            // Extract Centroids & Calculate Area
            let idCounter = 1;
            let sizeCategories = { 'Small (<500m²)': 0, 'Medium (500-2000m²)': 0, 'Large (>2000m²)': 0 };

            currentGeoJsonLayer.eachLayer(function(layer) {
                const center = layer.getBounds().getCenter();
                let areaSqMeters = 0;
                
                // Turf area
                if (layer.feature && layer.feature.geometry) {
                    areaSqMeters = turf.area(layer.feature);
                }

                if (areaSqMeters < 500) sizeCategories['Small (<500m²)']++;
                else if (areaSqMeters <= 2000) sizeCategories['Medium (500-2000m²)']++;
                else sizeCategories['Large (>2000m²)']++;

                reportData.push({
                    id: `SLUM-${idCounter.toString().padStart(3, '0')}`,
                    area: Math.round(areaSqMeters),
                    lat: center.lat.toFixed(5),
                    lng: center.lng.toFixed(5)
                });
                idCounter++;
            });

            renderChart(sizeCategories);

            // Cinematic Fly-To animation
            map.flyToBounds(bounds, {
                duration: 2.5,
                easeLinearity: 0.25
            });
        } catch(e) {
            console.log("Could not fly to bounds.");
        }

        // UI State: Success (Show Dashboard)
        dashboardStats.classList.remove('hidden');
        layerControls.classList.remove('hidden');
        actionButtons.classList.remove('hidden');
    }

    let slumChartInstance = null;
    function renderChart(categories) {
        const ctx = document.getElementById('slumChart').getContext('2d');
        if (slumChartInstance) slumChartInstance.destroy();
        
        slumChartInstance = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(categories),
                datasets: [{
                    data: Object.values(categories),
                    backgroundColor: ['#fca5a5', '#ef4444', '#991b1b'],
                    borderColor: 'rgba(0,0,0,0)',
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'right', labels: { color: '#94a3b8', font: { family: 'Inter' } } },
                    title: { display: true, text: 'Slum Size Distribution', color: '#f8fafc', font: { size: 16, family: 'Outfit' } }
                }
            }
        });
    }
});
