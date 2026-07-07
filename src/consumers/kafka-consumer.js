// Kafka Consumer
const { kafka, TOPIC_RAW_LISTINGS } = require('../../config/kafka-config');

const CONSUMER_GROUP_ID = process.env.KAFKA_CONSUMER_GROUP_ID || 'otometrik-csv-writer';

// raw-listings topic'ini dinler; her mesaj JSON.parse edilip onListing'e iletilir.
async function consumeListings(onListing, { groupId = CONSUMER_GROUP_ID } = {}) {
  const consumer = kafka.consumer({ groupId });
  await consumer.connect();
  await consumer.subscribe({ topic: TOPIC_RAW_LISTINGS, fromBeginning: true });

  await consumer.run({
    eachMessage: async ({ message }) => {
      if (!message.value) return;
      try {
        const listing = JSON.parse(message.value.toString());
        await onListing(listing);
      } catch (err) {
        console.error('Mesaj islenemedi:', err.message);
      }
    },
  });

  return consumer;
}

module.exports = { consumeListings, CONSUMER_GROUP_ID };
