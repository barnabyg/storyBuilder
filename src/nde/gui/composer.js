const draft = document.querySelector("#story-draft");
const picker = document.querySelector("#example-picker");

if (picker) {
  const choice = picker.elements.idea;
  const preview = document.querySelector("#example-preview");
  const dialog = document.querySelector("#replace-example");

  function showPremise() {
    preview.textContent = choice.selectedOptions[0]?.dataset.premise || "";
    picker.querySelector("button").disabled = !choice.value;
  }

  choice.addEventListener("change", showPremise);
  window.addEventListener("pageshow", showPremise);
  showPremise();
  picker.hidden = false;

  picker.addEventListener("submit", (event) => {
    const changed = Array.from(draft.querySelectorAll("input, textarea, select")).some((field) => {
      if (field.tagName === "SELECT") {
        const original = Array.from(field.options).find((option) => option.defaultSelected);
        return field.value !== (original || field.options[0]).value;
      }
      return field.value !== field.defaultValue;
    });
    if (changed || draft.dataset.edited === "true") {
      event.preventDefault();
      dialog.returnValue = "keep";
      dialog.showModal();
    }
  });

  dialog.addEventListener("close", () => {
    if (dialog.returnValue === "replace") picker.submit();
  });
}
