/**
 * Nuke Nodes Preview Widget
 * Handles displaying animated image sequence previews for NukeRead and NukeWrite nodes
 */

// Use new ComfyUI API with fallback for older versions
let app, api;
if (window?.comfyAPI?.app && window?.comfyAPI?.api) {
    // New API (ComfyUI frontend v1.28+)
    app = window.comfyAPI.app;
    api = window.comfyAPI.api;
} else {
    // Fallback for older ComfyUI versions
    const appModule = await import("../../scripts/app.js");
    const apiModule = await import("../../scripts/api.js");
    app = appModule.app;
    api = apiModule.api;
}

// Node types that support preview
const PREVIEW_NODE_TYPES = ["NukeRead", "NukeWrite"];

// Configuration
const PREVIEW_CONFIG = {
    maxWidth: 256,
    maxHeight: 256,
    padding: 10,
    fps: 8,  // Default playback FPS
    controlsHeight: 30,
};

/**
 * Create preview player with timeline controls
 */
function createPreviewPlayer(node) {
    const state = {
        images: [],
        currentFrame: 0,
        playing: false,
        fps: PREVIEW_CONFIG.fps,
        animationInterval: null,
    };

    // Preload all images
    const imageCache = [];

    function loadImage(index) {
        if (imageCache[index]) return imageCache[index];

        const imgData = state.images[index];
        if (!imgData) return null;

        const img = new Image();
        const url = api.apiURL(`/view?filename=${encodeURIComponent(imgData.filename)}&type=${imgData.type}&subfolder=${encodeURIComponent(imgData.subfolder || "")}&rand=${Math.random()}`);
        img.src = url;
        imageCache[index] = img;
        return img;
    }

    function setFrame(frameIndex) {
        if (!state.images.length) return;
        state.currentFrame = Math.max(0, Math.min(frameIndex, state.images.length - 1));
        node.setDirtyCanvas(true, true);
    }

    function play() {
        if (state.playing || state.images.length <= 1) return;
        state.playing = true;

        state.animationInterval = setInterval(() => {
            state.currentFrame = (state.currentFrame + 1) % state.images.length;
            node.setDirtyCanvas(true, true);
        }, 1000 / state.fps);
    }

    function pause() {
        state.playing = false;
        if (state.animationInterval) {
            clearInterval(state.animationInterval);
            state.animationInterval = null;
        }
    }

    function stop() {
        pause();
        setFrame(0);
    }

    function updateImages(images) {
        pause();
        state.images = images || [];
        state.currentFrame = 0;
        imageCache.length = 0;

        // Preload first few frames
        if (state.images.length > 0) {
            for (let i = 0; i < Math.min(5, state.images.length); i++) {
                loadImage(i);
            }
        }

        node.setDirtyCanvas(true, true);
    }

    function getCurrentImage() {
        return loadImage(state.currentFrame);
    }

    function cleanup() {
        pause();
        imageCache.length = 0;
        state.images = [];
    }

    return {
        state,
        setFrame,
        play,
        pause,
        stop,
        updateImages,
        getCurrentImage,
        cleanup,
    };
}

/**
 * Draw preview with timeline controls
 */
function drawPreview(node, ctx) {
    if (!node.previewPlayer || !node.previewPlayer.state.images.length) return;

    const player = node.previewPlayer;
    const state = player.state;

    // Check if show_preview is enabled
    let showPreview = true;
    if (node.widgets) {
        const previewWidget = node.widgets.find(w => w.name === "show_preview");
        if (previewWidget) {
            showPreview = previewWidget.value;
        }
    }

    if (!showPreview) return;

    // Calculate layout
    const widgetHeight = node.computeSize()[1] - (state.images.length > 1 ? PREVIEW_CONFIG.controlsHeight + 40 : 20);
    const availableWidth = node.size[0] - PREVIEW_CONFIG.padding * 2;
    const availableHeight = node.size[1] - widgetHeight - PREVIEW_CONFIG.padding;
    const controlsY = node.size[1] - PREVIEW_CONFIG.controlsHeight - 5;

    // Get current image
    const img = player.getCurrentImage();

    if (img && img.complete && img.naturalWidth > 0 && availableHeight > 50) {
        // Calculate scaled dimensions maintaining aspect ratio
        const scale = Math.min(
            availableWidth / img.naturalWidth,
            (availableHeight - (state.images.length > 1 ? PREVIEW_CONFIG.controlsHeight + 10 : 0)) / img.naturalHeight,
            1
        );

        const drawWidth = img.naturalWidth * scale;
        const drawHeight = img.naturalHeight * scale;
        const x = (node.size[0] - drawWidth) / 2;
        const y = widgetHeight + (availableHeight - drawHeight - (state.images.length > 1 ? PREVIEW_CONFIG.controlsHeight + 10 : 0)) / 2;

        // Draw background
        ctx.fillStyle = "#1a1a1a";
        ctx.fillRect(x - 2, y - 2, drawWidth + 4, drawHeight + 4);

        // Draw image
        ctx.drawImage(img, x, y, drawWidth, drawHeight);

        // Draw frame counter if multi-frame
        if (state.images.length > 1) {
            const frameText = `Frame ${state.currentFrame + 1} / ${state.images.length}`;
            ctx.fillStyle = "#888";
            ctx.font = "10px Arial";
            ctx.textAlign = "center";
            ctx.fillText(frameText, node.size[0] / 2, y + drawHeight + 12);

            // Draw timeline controls
            drawTimelineControls(ctx, node, controlsY);
        }
    }
}

/**
 * Draw timeline controls (play/pause, timeline scrubber)
 */
function drawTimelineControls(ctx, node, y) {
    const player = node.previewPlayer;
    const state = player.state;
    const controlsWidth = node.size[0] - PREVIEW_CONFIG.padding * 2;
    const x = PREVIEW_CONFIG.padding;

    // Background
    ctx.fillStyle = "#2a2a2a";
    ctx.fillRect(x, y, controlsWidth, PREVIEW_CONFIG.controlsHeight);

    // Play/Pause button
    const buttonSize = 20;
    const buttonX = x + 5;
    const buttonY = y + (PREVIEW_CONFIG.controlsHeight - buttonSize) / 2;

    ctx.fillStyle = state.playing ? "#4a4a4a" : "#3a3a3a";
    ctx.fillRect(buttonX, buttonY, buttonSize, buttonSize);

    ctx.fillStyle = "#ddd";
    ctx.font = "12px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(state.playing ? "⏸" : "▶", buttonX + buttonSize / 2, buttonY + buttonSize / 2);

    // Timeline scrubber
    const timelineX = buttonX + buttonSize + 10;
    const timelineWidth = controlsWidth - buttonSize - 20;
    const timelineY = y + (PREVIEW_CONFIG.controlsHeight - 4) / 2;

    // Timeline background
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(timelineX, timelineY, timelineWidth, 4);

    // Timeline progress
    const progress = state.images.length > 1 ? state.currentFrame / (state.images.length - 1) : 0;
    ctx.fillStyle = "#4a9eff";
    ctx.fillRect(timelineX, timelineY, timelineWidth * progress, 4);

    // Timeline handle
    const handleX = timelineX + timelineWidth * progress;
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(handleX, timelineY + 2, 6, 0, Math.PI * 2);
    ctx.fill();

    // Store hit areas for mouse interaction
    if (!node.previewHitAreas) {
        node.previewHitAreas = {};
    }
    node.previewHitAreas.playButton = { x: buttonX, y: buttonY, width: buttonSize, height: buttonSize };
    node.previewHitAreas.timeline = { x: timelineX, y: timelineY - 5, width: timelineWidth, height: 14 };

    ctx.textBaseline = "alphabetic";
}

/**
 * Handle mouse events for timeline controls
 */
function handleMouseEvent(node, event, pos) {
    if (!node.previewPlayer || !node.previewHitAreas) return false;

    const player = node.previewPlayer;
    const localPos = {
        x: pos[0],
        y: pos[1]
    };

    // Check play button
    const playBtn = node.previewHitAreas.playButton;
    if (localPos.x >= playBtn.x && localPos.x <= playBtn.x + playBtn.width &&
        localPos.y >= playBtn.y && localPos.y <= playBtn.y + playBtn.height) {
        if (event.type === "pointerdown") {
            if (player.state.playing) {
                player.pause();
            } else {
                player.play();
            }
            return true;
        }
    }

    // Check timeline
    const timeline = node.previewHitAreas.timeline;
    if (localPos.x >= timeline.x && localPos.x <= timeline.x + timeline.width &&
        localPos.y >= timeline.y && localPos.y <= timeline.y + timeline.height) {
        if (event.type === "pointerdown" || (event.type === "pointermove" && event.buttons === 1)) {
            const progress = Math.max(0, Math.min(1, (localPos.x - timeline.x) / timeline.width));
            const frame = Math.floor(progress * (player.state.images.length - 1));
            player.setFrame(frame);
            return true;
        }
    }

    return false;
}

// Register extension
app.registerExtension({
    name: "NukeNodes.Preview",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!PREVIEW_NODE_TYPES.includes(nodeData.name)) {
            return;
        }

        // Store original onExecuted
        const origOnExecuted = nodeType.prototype.onExecuted;

        nodeType.prototype.onExecuted = function(message) {
            console.log("[NukePreview] onExecuted called", message);

            // Handle preview images BEFORE calling original
            // This prevents ComfyUI from creating default image widgets
            if (message && message.images && message.images.length > 0) {
                console.log("[NukePreview] Found images:", message.images.length);

                if (!this.previewPlayer) {
                    this.previewPlayer = createPreviewPlayer(this);
                }
                this.previewPlayer.updateImages(message.images);

                // Check if show_preview is enabled
                let showPreview = true;
                if (this.widgets) {
                    const previewWidget = this.widgets.find(w => w.name === "show_preview");
                    if (previewWidget) {
                        showPreview = previewWidget.value;
                    }
                }

                console.log("[NukePreview] show_preview:", showPreview);

                // If preview is enabled, suppress default ComfyUI image display
                if (showPreview) {
                    // Store images in a custom property so we can still access them
                    this._nukePreviewImages = [...message.images];
                    // Clear images array to prevent default ComfyUI display
                    delete message.images;
                    message.images = [];

                    console.log("[NukePreview] Cleared message.images to suppress default display");
                }
            }

            // Call original if exists
            if (origOnExecuted) {
                origOnExecuted.apply(this, arguments);
            }

            // Force canvas redraw
            this.setDirtyCanvas(true, true);
        };

        // Store original onNodeCreated
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function() {
            if (origOnNodeCreated) {
                origOnNodeCreated.apply(this, arguments);
            }

            // Create preview player
            this.previewPlayer = createPreviewPlayer(this);

            // Override onDrawForeground to draw preview
            const origOnDrawForeground = this.onDrawForeground;
            this.onDrawForeground = function(ctx) {
                if (origOnDrawForeground) {
                    origOnDrawForeground.apply(this, arguments);
                }

                // Draw preview with timeline controls
                drawPreview(this, ctx);
            };

            // Handle mouse events
            const origOnMouseDown = this.onMouseDown;
            this.onMouseDown = function(event, pos, canvas) {
                if (handleMouseEvent(this, event, pos)) {
                    return true;
                }
                if (origOnMouseDown) {
                    return origOnMouseDown.call(this, event, pos, canvas);
                }
                return false;
            };

            const origOnMouseMove = this.onMouseMove;
            this.onMouseMove = function(event, pos, canvas) {
                if (handleMouseEvent(this, event, pos)) {
                    return true;
                }
                if (origOnMouseMove) {
                    return origOnMouseMove.call(this, event, pos, canvas);
                }
                return false;
            };

            // Resize node to fit preview
            const origComputeSize = this.computeSize;
            this.computeSize = function() {
                const size = origComputeSize ? origComputeSize.apply(this, arguments) : [200, 100];

                // Check if show_preview is enabled
                let showPreview = true;
                if (this.widgets) {
                    const previewWidget = this.widgets.find(w => w.name === "show_preview");
                    if (previewWidget) {
                        showPreview = previewWidget.value;
                    }
                }

                if (showPreview && this.previewPlayer && this.previewPlayer.state.images.length > 0) {
                    const isSequence = this.previewPlayer.state.images.length > 1;
                    const extraHeight = isSequence ? PREVIEW_CONFIG.controlsHeight + 40 : 20;
                    return [
                        Math.max(size[0], PREVIEW_CONFIG.maxWidth + PREVIEW_CONFIG.padding * 2),
                        size[1] + PREVIEW_CONFIG.maxHeight + PREVIEW_CONFIG.padding * 2 + extraHeight
                    ];
                }

                return size;
            };
        };

        // Clean up on node removal
        const origOnRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function() {
            if (origOnRemoved) {
                origOnRemoved.apply(this, arguments);
            }

            // Cleanup preview player
            if (this.previewPlayer) {
                this.previewPlayer.cleanup();
                this.previewPlayer = null;
            }
        };
    },
});

console.log("[NukeNodes] Preview extension loaded");
