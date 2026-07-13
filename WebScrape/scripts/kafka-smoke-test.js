// Kafka smoke test: verifies producer/consumer connectivity end-to-end
const { kafka } = require('../config/kafka-config');

const SMOKE_TEST_TOPIC = 'otometrik-smoke-test';

async function run() {
  const producer = kafka.producer();
  const consumer = kafka.consumer({ groupId: `smoke-test-${Date.now()}` });

  await producer.connect();
  await consumer.connect();
  await consumer.subscribe({ topic: SMOKE_TEST_TOPIC, fromBeginning: true });

  const testMessage = { ping: 'otometrik-smoke-test', ts: new Date().toISOString() };

  const received = new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('Smoke test timed out waiting for message')), 15000);
    consumer.run({
      eachMessage: async ({ message }) => {
        clearTimeout(timeout);
        resolve(JSON.parse(message.value.toString()));
      },
    });
  });

  await producer.send({
    topic: SMOKE_TEST_TOPIC,
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
