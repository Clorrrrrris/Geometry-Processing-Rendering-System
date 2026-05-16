# Assignment 3 — Mesh Simplification System

A geometry-processing system implementing a half-edge mesh data structure and quadric-error mesh simplification for real-time level-of-detail (LOD) rendering.

## Overview

This project implements a complete half-edge mesh processing pipeline for triangular manifold meshes loaded from OBJ files. The system supports interactive mesh traversal, topology-aware edge collapse, undo/redo operations, and quadric-error–based mesh simplification.

The implementation focuses on scalable geometry rendering and dynamic level-of-detail control while preserving mesh topology and visual fidelity.

## Tech Stack

- Python

- OpenGL

- GLSL

- NumPy

- trimesh

- sortedcontainers

## Result

The system supports progressive mesh simplification from full-resolution meshes down to coarse low-polygon representations while maintaining interactive rendering and stable topology updates.

# Assignment 4 — Parallel Ray Tracing System

A CPU/GPU-accelerated ray tracing renderer built with Taichi, supporting hierarchical scene graphs, shading, shadows, reflections, and triangle-mesh rendering.

## Overview

This project implements a physically inspired ray tracing pipeline capable of rendering complex scenes defined through JSON scene descriptions. The renderer supports multiple geometry types, recursive ray interactions, hierarchical transformations, and parallel rendering acceleration using Taichi.

The system emphasizes scalable rendering performance, modular scene representation, and extensible rendering architecture.

## Tech Stack

- Python

- Taichi

- OpenGL

- GLSL

- JSON5

- NumPy

- trimesh

## Result

The renderer supports interactive and scalable rendering of complex scenes with recursive lighting effects, hierarchical geometry, and GPU-accelerated computation.

