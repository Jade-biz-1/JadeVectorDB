// Mock next/head — return null so <title> tags don't pollute the jsdom body
// and cause "multiple elements found" errors in tests.
const Head = () => null;
Head.displayName = 'MockNextHead';
module.exports = Head;
module.exports.default = Head;
