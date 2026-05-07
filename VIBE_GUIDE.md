# Vibe Coding Collaboration Guide

Welcome to the **Vibe Vision Assistant** project. This repo is designed to be "vibe-coded" — meaning it thrives on high-level intent, shared context, and rapid iteration across multiple AI agents and human collaborators.

## The Workflow

### 1. The Source of Truth: `VIBE_GUIDE.md`
Always update this file if the project's direction or "vibe" shifts. Agents should read this first.

### 2. Context Syncing (`/prompts` & `/docs`)
Since different agents (Antigravity, Cursor, Replit, etc.) have different memory spans, we keep a `prompts/` directory. 
- **System Prompts**: The "personality" of the assistant.
- **Architectural Vibes**: High-level sketches of how things should work.

### 3. Agent-Friendly Commits
When an agent (or human) makes a change, use descriptive commit messages that explain the *intent* (the vibe) rather than just the code change.
Example: `feat: add spatial awareness vibe to the vision module`

### 4. Branching Strategy
- `main`: Stable(ish) "vibes".
- `feature/[vibe-name]`: Experimental features.

## Collaboration with Friends
1. **Invite your friend** to the GitHub repository.
2. **Use Issues for "Vibe Requests"**: Instead of strict specs, describe what you want the assistant to *feel* like or *do*.
3. **Agent Hand-offs**: If you move from one agent to another, tell the new agent to "Read the README and VIBE_GUIDE to pick up the signal."

## Tech Stack
- **Vision**: (Planned) OpenCV, Mediapipe, or Transformers.
- **Brain**: Agentic framework (to be decided).
- **Vibe**: 100% pure intent.
