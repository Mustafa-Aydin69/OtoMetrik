// Kafka smoke test: verifies producer/consumer connectivity end-to-end
const { kafka, TOPIC_RAW_LISTINGS } = require('../config/kafka-config');

async function run() {
  const producer = kafka.producer();
  const consumer = kafka.consumer({ groupId: 'smoke-test-group' });

  await producer.connect();
  await consumer.connect();
  await consumer.subscribe({ topic: TOPIC_RAW_LISTINGS, fromBeginning: false });

  const testMessage = { ping: 'otometrik-smoke-test', ts: new Date().toISOString() };

  const received = new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('Smoke test timed out waiting for message')), 10000);
    consumer.run({
      eachMessage: async ({ message }) => {
        clearTimeout(timeout);
        resolve(JSON.parse(message.value.toString()));
      },
    });
  });

  await producer.send({
    topic: TOPIC_RAW_LISTINGS,
    messages: [{ value: JSON.stringify(testMessage) }],
  });

  const result = await received;
  console.log('Kafka smoke test OK:', result);

  await producer.disconnect();
  await consumer.disconnect();
  process.exit(0);
}

run().catch((err) => {
  console.error('Kafka smoke test FAILED:', err.message);
  process.exit(1);
});
