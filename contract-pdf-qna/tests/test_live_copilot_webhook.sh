#!/bin/bash
# Test script for Live Copilot Webhook
# This script sends sample transcripts to test the copilot integration

BASE_URL="${1:-http://localhost:8001}"

echo "=== Testing Live Copilot Webhook Integration ==="
echo "Backend URL: $BASE_URL"
echo ""

# Test 1: Customer identification with phone number
echo "📞 Test 1: Customer provides phone number"
curl -X POST "$BASE_URL/webhook" \
  -H "Content-Type: application/json" \
  -d '{
    "sessionId": "test-session-001",
    "contactId": "test-contact-001",
    "speaker": "CUSTOMER",
    "text": "Hi, my name is John Smith. My phone number is 555-123-4567.",
    "phone": "5551234567",
    "state": "Texas",
    "contractType": "RE",
    "plan": "ShieldPlus",
    "isPartial": false,
    "beginOffsetMillis": 0,
    "endOffsetMillis": 5000
  }'
echo -e "\n"

sleep 2

# Test 2: Customer asks coverage question
echo "❓ Test 2: Customer asks coverage question"
curl -X POST "$BASE_URL/webhook" \
  -H "Content-Type: application/json" \
  -d '{
    "sessionId": "test-session-001",
    "contactId": "test-contact-001",
    "speaker": "CUSTOMER",
    "text": "My water heater is leaking. Is that covered under my plan?",
    "phone": "5551234567",
    "state": "Texas",
    "contractType": "RE",
    "plan": "ShieldPlus",
    "isPartial": false,
    "beginOffsetMillis": 5000,
    "endOffsetMillis": 10000
  }'
echo -e "\n"

sleep 2

# Test 3: Customer describes appliance problem
echo "🔧 Test 3: Customer describes appliance problem"
curl -X POST "$BASE_URL/webhook" \
  -H "Content-Type: application/json" \
  -d '{
    "sessionId": "test-session-001",
    "contactId": "test-contact-001",
    "speaker": "CUSTOMER",
    "text": "The refrigerator stopped cooling yesterday. Everything inside is getting warm.",
    "phone": "5551234567",
    "state": "Texas",
    "contractType": "RE",
    "plan": "ShieldPlus",
    "isPartial": false,
    "beginOffsetMillis": 10000,
    "endOffsetMillis": 15000
  }'
echo -e "\n"

sleep 2

# Test 4: Customer asks about limits
echo "💰 Test 4: Customer asks about coverage limits"
curl -X POST "$BASE_URL/webhook" \
  -H "Content-Type: application/json" \
  -d '{
    "sessionId": "test-session-001",
    "contactId": "test-contact-001",
    "speaker": "CUSTOMER",
    "text": "What is the maximum coverage limit for HVAC repairs?",
    "phone": "5551234567",
    "state": "Texas",
    "contractType": "RE",
    "plan": "ShieldPlus",
    "isPartial": false,
    "beginOffsetMillis": 15000,
    "endOffsetMillis": 20000
  }'
echo -e "\n"

echo "=== Tests Complete ==="
echo ""
echo "Check the backend logs for:"
echo "  🔴 TRANSCRIPT RECEIVED - Each transcript logged"
echo "  🟢 COPILOT SUGGESTION - AI suggestions generated"
echo ""
echo "Check the frontend Analyze Live UI for:"
echo "  - User Details card populated"
echo "  - AI Suggestions panel showing responses"

