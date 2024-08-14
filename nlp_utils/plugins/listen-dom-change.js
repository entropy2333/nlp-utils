// ==UserScript==
// @name         Listen Dom Change
// @namespace    http://tampermonkey.net/
// @version      2024-06-21
// @description  try to take over the world!
// @author       You
// @match        http:/example.com/
// @icon         https://www.google.com/s2/favicons?sz=64&domain=baosteel.com
// @grant        none
// ==/UserScript==

(function() {
    'use strict';

    // Selector: Replace with the actual selectors for the complete button and target button
    const completeButtonSelector = '#main > div > div:nth-child(5) > div > div.video_con > div.player.playerOne > div.video-wrapper > div.video-content > div.video-modal.show > div > div:nth-child(6) > div > button.go-next';

    const targetButtonSelector = completeButtonSelector;

    // Use MutationObserver to monitor DOM changes
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            // Check if the complete button exists
            const completeButton = document.querySelector(completeButtonSelector);
            if (completeButton) {
                // Find the target button and click it
                const targetButton = document.querySelector(targetButtonSelector);
                if (targetButton) {
                    console.log("go next!");
                    setTimeout(() => { targetButton.click(); }, 100); // Delay the click to avoid interfering with user actions
                    targetButton.click();
                }
                // Stop observing after the click (uncomment if needed)
                // observer.disconnect();
            }
        });
    });

    // Start observing the root node of the document
    observer.observe(document, { childList: true, subtree: true });

    // Optional: Check once if the complete button exists when the page loads
    /*
    window.addEventListener('load', () => {
        const completeButton = document.querySelector(completeButtonSelector);
        if (completeButton) {
            const targetButton = document.querySelector(targetButtonSelector);
            if (targetButton) {
                targetButton.click();
            }
        }
    });
    */
})();
