// Kafka configurations
require('dotenv').config();
const { Kafka } = require('kafkajs');

const kafka = new Kafka({
  clientId: process.env.KAFKA_CLIENT_ID || 'otometrik',
  brokers: [process.env.KAFKA_BROKER || 'localhost:9092'],
});

const TOPIC_RAW_LISTINGS = process.env.KAFKA_TOPIC_RAW_LISTINGS || 'raw-listings';

module.exports = { kafka, TOPIC_RAW_LISTINGS };
