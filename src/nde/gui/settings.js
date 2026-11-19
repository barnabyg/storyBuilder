(() => {
  const model = document.querySelector("#model");
  const effort = document.querySelector("#effort");
  const save = document.querySelector("#openai-settings button");
  const levels = document.querySelector("#thinking-levels").content.querySelectorAll("option");

  function updateThinkingLevels() {
    const selected = model.selectedOptions[0];
    const supported = (selected.dataset.efforts || "").split(" ");
    const options = Array.from(levels).filter(option => supported.includes(option.value));
    const previous = effort.value;
    const fallback = options.length ? "Model default" : "Unavailable";
    effort.replaceChildren(new Option(fallback, ""), ...options.map(option => option.cloneNode(true)));
    if (supported.includes(previous)) effort.value = previous;
    effort.disabled = options.length === 0;
    save.disabled = selected.disabled;
  }

  model.addEventListener("change", updateThinkingLevels);
})();
