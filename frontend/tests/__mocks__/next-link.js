// Global mock for next/link — renders a plain <a> tag
const React = require('react');
const Link = ({ children, href, ...props }) =>
  React.createElement('a', { href, ...props }, children);
Link.displayName = 'MockNextLink';
module.exports = Link;
module.exports.default = Link;
