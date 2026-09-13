// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract HealthcareAccessRegistry {
    event HealthcareEvent(
        uint256 indexed eventId,
        address indexed entity,
        string role,
        string action,
        bool sensitiveRecordAccessed,
        bool roleMismatch,
        bool credentialRevocation,
        bool geolocationMismatch,
        bool issuerVerifierCollusion,
        uint256 accessFrequency,
        uint256 sessionDurationSeconds,
        uint256 timestamp
    );

    uint256 public eventCount;
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    function recordEvent(
        string calldata role,
        string calldata action,
        bool sensitiveRecordAccessed,
        bool roleMismatch,
        bool credentialRevocation,
        bool geolocationMismatch,
        bool issuerVerifierCollusion,
        uint256 accessFrequency,
        uint256 sessionDurationSeconds
    ) external returns (uint256) {
        eventCount += 1;
        emit HealthcareEvent(
            eventCount,
            msg.sender,
            role,
            action,
            sensitiveRecordAccessed,
            roleMismatch,
            credentialRevocation,
            geolocationMismatch,
            issuerVerifierCollusion,
            accessFrequency,
            sessionDurationSeconds,
            block.timestamp
        );
        return eventCount;
    }

    function totalEvents() external view returns (uint256) {
        return eventCount;
    }
}
