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

const notice = document.querySelector("#busy-notice");
const activeStory = document.querySelector("#active-story");
const writeButton = document.querySelector("#write-story");
writeButton.disabled = !notice.hidden;

// Update only the submit controls so editing and keyboard focus survive a status check.
let checking = false;
async function checkAvailability() {
  if (document.hidden || checking) return;
  checking = true;
  try {
    const response = await fetch("/writing-status", { cache: "no-store" });
    if (!response.ok) return;
    const { job } = await response.json();
    if (job) {
      const url = "/jobs/" + encodeURIComponent(job);
      notice.querySelector("a").href = url;
      activeStory.querySelector("a").href = url;
    }
    notice.hidden = !job;
    activeStory.hidden = !job;
    writeButton.disabled = Boolean(job);
  } catch {
    // The next poll retries. The server also checks availability on every submission.
  } finally {
    checking = false;
  }
}
window.addEventListener("pageshow", checkAvailability);
document.addEventListener("visibilitychange", checkAvailability);
window.setInterval(checkAvailability, 3000);
