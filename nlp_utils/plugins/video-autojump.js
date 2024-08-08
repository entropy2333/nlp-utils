// ==UserScript==
// @name         Autojump
// @namespace    http://tampermonkey.net/
// @version      2024-06-21
// @description  Automatically clicks target button when video ends
// @author       You
// @match        https://example.com/
// @icon         
// @grant        none
// ==/UserScript==

(function() {
    'use strict';

    const jwplayer_video_selector = '#myPlayer > div.jw-media.jw-reset > video';
    const target_button_selector = '#app > div > section > div > div.left-box > span > a';

    function addVideoEventListeners() {
        const jwplayer_video = document.querySelector(jwplayer_video_selector);
        if (jwplayer_video) {
            jwplayer_video.addEventListener('play', function() {
                console.log('Video started playing');
            });
            jwplayer_video.addEventListener('ended', function() {
                const target_button = document.querySelector(target_button_selector);
                if (target_button) {
                    target_button.click();
                    console.log('Video has finished playing, button clicked');
                } else {
                    console.log('Target button not found');
                }
            });
        } else {
            console.log('Fail to find video');
        }
    }

    function waitForVideo() {
        const observer = new MutationObserver((mutations, obs) => {
            if (document.querySelector(jwplayer_video_selector)) {
                addVideoEventListeners();
                obs.disconnect();
            }
        });

        observer.observe(document, {
            childList: true,
            subtree: true
        });
    }

    window.addEventListener('load', () => {
        setTimeout(waitForVideo, 1000); // Delay to ensure elements are loaded
    });
})();
