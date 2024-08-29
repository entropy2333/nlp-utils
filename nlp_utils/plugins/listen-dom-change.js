// ==UserScript==
// @name         Listen Dom Change
// @namespace    http://tampermonkey.net/
// @version      2024-08-29
// @description  try to take over the world!
// @author       You
// @match        http://example.com/
// @icon         https://www.google.com/s2/favicons?sz=64&domain=baosteel.com
// @run-at       document-end
// @grant        none
// ==/UserScript==

(function() {
    'use strict';

    // Selector: Replace with the actual selectors for the complete button and target button
    const completeButtonSelector = '';
    const targetButtonSelector = '';
    const saveButtonSelector = '';

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

    // Click the save button
    function clickSaveButton() {
        const saveButton = document.querySelector(saveButtonSelector);
        if (saveButton) {
            console.log("Clicking the save button...");
            saveButton.click();
        } else {
            console.log("Save button not found.");
        }
    }

    // Start observing the root node of the document
    window.addEventListener('load', () => {
        window.alert=function(){};
        console.log('set alert to null');
        
        observer.observe(document, { childList: true, subtree: true });
        console.log('start listening');

        // Click the save button every 20 minutes
        setInterval(clickSaveButton, 20 * 60 * 1000);
    });
})();
