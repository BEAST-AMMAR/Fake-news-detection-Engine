# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.1.x   | :white_check_mark: |
| < 2.0   | :x:                |

## Security Architecture & Defenses

Project Beta incorporates multiple layers of operational security to protect inference environments:

1. **Server-Side Request Forgery (SSRF) Prevention:**
   The URL scraping module (`src/inference/scraper.py`) enforces strict scheme validation (`http`, `https`) and executes pre-flight DNS address resolution to reject:
   - Loopback interfaces (`127.0.0.0/8`, `::1`)
   - RFC 1918 private subnets (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`)
   - Link-local and cloud metadata addresses (`169.254.169.254`, `fe80::/10`)
   - Multicast and reserved IP ranges

2. **Denial of Service (DoS) Mitigation:**
   - Input payloads on `/api/verify` are constrained by strict length limits (`max_length=50,000` chars for text, `1,000` chars for headlines).
   - Scraper network requests timeout after $12.0\text{ seconds}$ with bounded response sizes.

3. **Data Leakage & Model Integrity:**
   - Publisher attribution headers and datelines (`(Reuters) - `, etc.) are stripped during feature extraction to prevent shortcut learning and brand bias.
   - External ClaimReview and RSS consensus queries are cached with a $10\text{ minute}$ in-memory TTL to avoid upstream rate-limiting.

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please do not open a public issue. Instead, report it privately to:

**Security Contact:** `muhammada.23.beai@acharya.ac.in`

Please provide:
- A description of the issue and potential impact
- Steps to reproduce or proof-of-concept payload
- Environment details

Reports will be acknowledged within 48 hours, and a remediation patch will be released promptly.
