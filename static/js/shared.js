/**
 * Auto-trigger the first form on the page after a delay.
 * @param {number} delay - Milliseconds to wait after page load before submitting.
 */
function autoTriggerForm(delay) {
    window.addEventListener("load", function () {
        setTimeout(function () {
            document
                .querySelector("form")
                .dispatchEvent(new Event("submit", { bubbles: true }));
        }, delay);
    });
}
