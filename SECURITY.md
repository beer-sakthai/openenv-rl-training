# Security Policy

## Supported Versions

Use this section to tell people about which versions of your project are
currently being supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

Please report (suspected) security vulnerabilities to our team. You can do this by submitting an issue to this repository, clearly marked with "[Security]" in the title.
If the vulnerability is critical, please refrain from sharing sensitive details in the public issue. We will reach out to you directly to establish a secure communication channel.

We aim to acknowledge receipt of vulnerability reports within 48 hours and provide regular updates on the resolution progress.

## Sandboxing Architecture

This repository contains components that execute code in sandboxed environments (`openenv-custom-training/agent_tools/server`). While these environments are designed to isolate execution, they are intended for use in controlled, internal training workflows. Please ensure that untrusted inputs are not passed directly to these environments outside of their intended scope.
