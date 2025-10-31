// static/app.js
async function init() {
  const res = await fetch("/get_options");
  const options = await res.json();
  const fieldsDiv = document.getElementById("fields");
  window._features = Object.keys(options);

  // Build inputs
  for (const col of window._features) {
    const container = document.createElement("div");
    container.className = "field";
    const label = document.createElement("label");
    label.textContent = col;
    container.appendChild(label);

    const vals = options[col] || [];
    if (vals.length > 0 && vals.length <= 200) {
      const select = document.createElement("select");
      select.name = col;
      const empty = document.createElement("option");
      empty.value = "";
      empty.textContent = "-- select --";
      select.appendChild(empty);
      for (const v of vals) {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        select.appendChild(opt);
      }
      container.appendChild(select);
    } else {
      const input = document.createElement("input");
      input.type = "text";
      input.name = col;
      input.placeholder = "Enter " + col;
      container.appendChild(input);
    }
    fieldsDiv.appendChild(container);
  }

  document.getElementById("predict-btn").addEventListener("click", async () => {
    const form = document.getElementById("predict-form");
    const data = {};
    for (const el of form.elements) {
      if (!el.name) continue;
      data[el.name] = el.value;
    }
    document.getElementById("predict-btn").disabled = true;
    document.getElementById("predict-btn").textContent = "Predicting...";
    try {
      const r = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      });
      const j = await r.json();
      document.getElementById("pred-text").textContent = `${j.target} ≈ ${Number(j.prediction).toFixed(2)}`;
      document.getElementById("result").style.display = "block";
    } catch (e) {
      alert("Prediction failed: " + e);
    } finally {
      document.getElementById("predict-btn").disabled = false;
      document.getElementById("predict-btn").textContent = "Predict";
    }
  });
}

window.addEventListener("load", init);
