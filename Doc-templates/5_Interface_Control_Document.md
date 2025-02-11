# Interface Control Document (ICD)

## Document Information

- **Title:** Interface Control Document
- **Project:** Data-Driven Greenhouse Climate Control System
- **Version:** 1.0
- **Last Updated:** [Date]

---

## 1. Introduction

### 1.1 Purpose

This document specifies the interfaces among the internal components of the greenhouse control system, as well as between the system and external entities. It ensures that all subsystems communicate in a consistent, secure, and interoperable manner.

### 1.2 Scope

The ICD covers:

- All external interfaces (e.g., weather services, energy systems)
- All internal interfaces (e.g., between sensors, actuators, APIs, and data stores)
- Data formats, protocols, and error handling mechanisms used for interface communication

---

## 2. Interface Overview

### 2.1 System Context Diagram

```
[External Systems]
      ↕
 [API Gateway]
      ↕
 [Core System]
      ↕
[Hardware Interface]
      ↕
[Sensors / Actuators]
```

### 2.2 Interface Categories

1. Sensor Interfaces
2. Actuator Interfaces
3. External System APIs
4. User Interfaces
5. Database Interfaces

---

## 3. Sensor Network Interface

### 3.1 Temperature Sensor Interface

```json
{
  "endpoint": "/api/v1/sensors/temperature",
  "method": "POST",
  "payload": {
    "sensor_id": "string",
    "timestamp": "ISO8601",
    "temperature": "float",
    "unit": "string",
    "status": "string"
  },
  "response": {
    "status": "string",
    "message": "string"
  }
}
```

### 3.2 Humidity Sensor Interface

```json
{
  "endpoint": "/api/v1/sensors/humidity",
  "method": "POST",
  "payload": {
    "sensor_id": "string",
    "timestamp": "ISO8601",
    "humidity": "float",
    "unit": "string",
    "status": "string"
  }
}
```

---

## 4. Actuator Control Interface

### 4.1 Ventilation Control

```json
{
  "endpoint": "/api/v1/actuators/ventilation",
  "method": "PUT",
  "payload": {
    "actuator_id": "string",
    "command": "string",
    "parameters": {
      "speed": "integer",
      "direction": "string"
    }
  }
}
```

### 4.2 Irrigation Control

```json
{
  "endpoint": "/api/v1/actuators/irrigation",
  "method": "PUT",
  "payload": {
    "zone_id": "string",
    "duration": "integer",
    "flow_rate": "float"
  }
}
```

---

## 5. External System Interfaces

### 5.1 Weather Service API

```json
{
  "endpoint": "/api/v1/weather",
  "method": "GET",
  "parameters": {
    "location": "string",
    "forecast_hours": "integer"
  },
  "response": {
    "temperature": "float",
    "humidity": "float",
    "precipitation": "float",
    "forecast": "array"
  }
}
```

### 5.2 Energy Management System

```json
{
  "endpoint": "/api/v1/energy",
  "method": "GET",
  "response": {
    "current_usage": "float",
    "peak_hours": "array",
    "pricing": "object"
  }
}
```

---

## 6. Data Exchange Formats

### 6.1 Sensor Data Format

```json
{
  "version": "1.0",
  "sensor_data": {
    "id": "string",
    "type": "string",
    "value": "float",
    "unit": "string",
    "timestamp": "ISO8601",
    "metadata": {
      "location": "string",
      "calibration": "object"
    }
  }
}
```

### 6.2 Control Command Format

```json
{
  "version": "1.0",
  "command": {
    "id": "string",
    "type": "string",
    "parameters": "object",
    "priority": "integer",
    "timestamp": "ISO8601"
  }
}
```

---

## 7. Communication Protocols

### 7.1 Real-Time Data via MQTT

- **Protocol:** MQTT  
- **QoS Levels:** 0, 1, 2  
- **Topic Structure:** `greenhouse/{zone_id}/{sensor_type}`

### 7.2 REST API

- **Authentication:** OAuth 2.0  
- **Content-Type:** application/json  
- **Rate Limiting:** 1,000 requests per minute

### 7.3 WebSocket

- **Protocol:** WSS  
- **Heartbeat:** Every 30 seconds  
- **Reconnection Strategy:** Exponential backoff

---

## 8. Error Handling

### 8.1 Error Codes

```json
{
  "4xx": {
    "400": "Bad Request",
    "401": "Unauthorized",
    "403": "Forbidden",
    "404": "Not Found",
    "429": "Too Many Requests"
  },
  "5xx": {
    "500": "Internal Server Error",
    "503": "Service Unavailable"
  }
}
```

### 8.2 Error Response Format

```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": "object",
    "timestamp": "ISO8601"
  }
}
```

---

## 9. Security

### 9.1 Authentication

- **Mechanism:** OAuth 2.0 flows with API key management and token refresh procedures.

### 9.2 Authorization

- **Mechanism:** Role-based access control, with defined scopes and a permission hierarchy to manage access rights.

---

## 10. Performance Requirements

### 10.1 Response Times

- **Sensor Data:** < 100 ms
- **Control Commands:** < 50 ms
- **Analytics Queries:** < 1 second

### 10.2 Throughput

- **Sensor Readings:** 1,000 per second
- **Control Commands:** 100 per second
- **API Requests:** 1,000 per minute

---

## 11. Monitoring and Logging

### 11.1 Metrics Monitored

- Response times across interfaces
- Error rates and types
- Request volumes
- Success and failure statistics

### 11.2 Log Format

```json
{
  "timestamp": "ISO8601",
  "level": "string",
  "service": "string",
  "message": "string",
  "metadata": "object"
}
```

---

## Appendices

### Appendix A: API Documentation

Detailed API usage and endpoint documentation.

### Appendix B: Message Formats

Full specifications for message payloads and schema definitions.

### Appendix C: Error Code Reference

Comprehensive list of error codes and their meanings.

### Appendix D: Security Protocols

Documentation of the security measures, protocols, and best practices used in this system.
