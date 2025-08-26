# Example Workflows

This directory contains example ComfyUI workflows demonstrating the use of Nuke nodes.

## Basic Compositing Workflow

This example shows how to use the merge nodes for basic compositing:

1. Load two images
2. Use NukeGrade to color correct each image
3. Use NukeTransform to position/scale images
4. Use NukeMerge to composite with "over" operation
5. Apply NukeBlur for final polish

## Color Grading Workflow

Professional color grading pipeline:

1. Load base image
2. NukeLevels for basic exposure correction
3. NukeGrade for lift/gamma/gain adjustment
4. NukeColorCorrect for creative color styling
5. Use masks for selective corrections

## Motion Graphics Workflow

Creating motion blur and transform effects:

1. Load static image
2. NukeTransform with motion blur enabled
3. NukeMotionBlur for additional directional blur
4. NukeMerge to composite multiple motion elements

## Advanced Compositing

Complex multi-layer compositing:

1. Background preparation with NukeGrade
2. Foreground element with NukeTransform
3. NukeCornerPin for perspective matching
4. NukeDefocus for depth integration
5. Final NukeMerge with custom blend modes

## Installation Note

To use these workflows:

1. Install Nuke Nodes for ComfyUI
2. Load the workflow JSON files into ComfyUI
3. Replace placeholder images with your own content
4. Adjust parameters as needed

Each workflow includes detailed node comments explaining the purpose and settings.
