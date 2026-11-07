# Security Policy

## Supported Versions
Use this section to tell people about which versions of your project are currently being supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability
Please do not report security vulnerabilities through public GitHub issues. Instead, send an email to security@yourdomain.com. You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

## Security Best Practices in this Project
- **API Keys:** Never commit your `.env` file. The `.gitignore` is configured to exclude it.
- **Authentication:** Passwords are hashed using `bcrypt`. Sessions are secured using JWT with expiration times.
- **Database:** SQL Injection is prevented by using SQLAlchemy ORM.
- **Data Privacy:** Uploaded datasets are stored locally in the `storage/` directory and are not sent to any external servers, except to the LLM for schema inspection (if using a cloud LLM).