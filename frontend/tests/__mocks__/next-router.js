// Global mock for next/router
const mockPush = jest.fn();
const mockReplace = jest.fn();
const mockBack = jest.fn();

const useRouter = jest.fn(() => ({
  push: mockPush,
  replace: mockReplace,
  back: mockBack,
  prefetch: jest.fn(),
  query: {},
  pathname: '/',
  asPath: '/',
  route: '/',
  events: {
    on: jest.fn(),
    off: jest.fn(),
    emit: jest.fn(),
  },
}));

module.exports = { useRouter, default: { useRouter } };
