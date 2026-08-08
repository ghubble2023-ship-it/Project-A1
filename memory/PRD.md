# Project AI — Core Scanner (Mobile App)

## Purpose
A companion mobile app for the **Project AI Hardened Core v1** engine — a rule-based catfish / romance-scam detection tool. Users run individual or comprehensive scans on suspicious contacts and receive a forensic-style verdict (LOW / MEDIUM / HIGH risk) backed by evidence categories.

## Core Features
- **Text Analysis** — paste chat + profile text; the engine detects love bombing, off-platform migration, unverifiable identity, pressure/urgency, and isolation. Money-only requests do not auto-escalate.
- **Photo Scan** — 7-flag manual checklist (contact shadow, glasses, lighting, skin texture, edge blending, catchlights, background) plus observation quality (good / partial / poor) that dampens verdicts when poor.
- **Profile Check** — followers, following, posts, photos, account age, bio, and "they contacted first" toggle → LOW / MEDIUM / ELEVATED concern.
- **Phone Lookup** — number validation, region hint, area-code vs claimed-location cross-check.
- **Full Scan** — combines all four modules into a single verdict dashboard; overall risk is text-driven per the engine contract.
- **Archive** — server-side scan history, tap to review any past verdict; clear-all supported.

## Engine Contract
Implemented in `backend/project_ai_core.py`, exposed via FastAPI at `/api/scan/{text,photo,profile,phone,full}` and `/api/scans` (list/save/get/delete/clear). Every function passes the fuzz + fixture assertions from the user-supplied `test_project_ai_core.py`. Engine name: `Project AI Hardened Core v1`.

## Design
Brutalist mobile personality: stark black/white, hard 2pt borders, zero border-radius, monospaced (IBM Plex Mono) + display (Space Grotesk) type stack, saturated risk color hero blocks (Signal Red / Warning Yellow / Safe Green).

## Privacy
Only scan verdicts and evidence categories are stored — never the raw chat text, profile content, photos, or phone number bodies.

## Business Enhancement (single, high-leverage)
The archive doubles as a **shareable evidence bundle**: each scan gets a stable id + hero verdict image, so a user can later reference "Scan #a3f2" when talking to a friend or a moderator — turning the app from a private tool into a low-friction warning network.
