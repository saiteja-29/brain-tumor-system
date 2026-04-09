async function uploadImage() {

    const input = document.getElementById("imageInput");
    const file = input.files[0];

    if (!file) {
        alert("Please select an image");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    // Show prediction
    document.getElementById("result").innerText =
        `Prediction: ${data.prediction} | Confidence: ${(data.confidence * 100).toFixed(2)}%`;

    // Show heatmap
    const heatmapImg = document.getElementById("heatmap");
    heatmapImg.src = "data:image/jpeg;base64," + data.heatmap;
}