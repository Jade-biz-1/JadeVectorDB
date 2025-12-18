/**
 * Component tests for Badge.jsx
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import Badge from '../components/tutorial/Badge';

describe('Badge Component', () => {
  const mockAchievement = {
    id: 'test-achievement',
    name: 'Test Achievement',
    description: 'A test achievement',
    icon: '🏆',
    tier: 'gold',
    points: 50
  };

  describe('Rendering', () => {
    it('should render unlocked achievement', () => {
      render(
        <Badge
          achievement={mockAchievement}
          unlocked={true}
          showDetails={true}
        />
      );

      expect(screen.getByText(mockAchievement.name)).toBeInTheDocument();
      expect(screen.getByText(mockAchievement.icon)).toBeInTheDocument();
      expect(screen.getByText('+50 pts')).toBeInTheDocument();
    });

    it('should render locked achievement with lock icon', () => {
      render(
        <Badge
          achievement={mockAchievement}
          unlocked={false}
          showDetails={true}
        />
      );

      expect(screen.getByText('🔒')).toBeInTheDocument();
      expect(screen.getByText(mockAchievement.name)).toBeInTheDocument();
    });

    it('should display unlock date when provided', () => {
      const unlockedAt = new Date('2023-01-15').toISOString();

      render(
        <Badge
          achievement={mockAchievement}
          unlocked={true}
          unlockedAt={unlockedAt}
          showDetails={true}
        />
      );

      expect(screen.getByText(/1\/15\/2023/)).toBeInTheDocument();
    });

    it('should hide details when showDetails is false', () => {
      render(
        <Badge
          achievement={mockAchievement}
          unlocked={true}
          showDetails={false}
        />
      );

      expect(screen.queryByText(mockAchievement.name)).not.toBeInTheDocument();
      expect(screen.queryByText('+50 pts')).not.toBeInTheDocument();
    });
  });

  describe('Size Variants', () => {
    it('should render small size', () => {
      const { container } = render(
        <Badge
          achievement={mockAchievement}
          unlocked={true}
          size="small"
        />
      );

      const badgeElement = container.querySelector('.w-16');
      expect(badgeElement).toBeInTheDocument();
    });

    it('should render medium size (default)', () => {
      const { container } = render(
        <Badge
          achievement={mockAchievement}
          unlocked={true}
          size="medium"
        />
      );

      const badgeElement = container.querySelector('.w-24');
      expect(badgeElement).toBeInTheDocument();
    });

    it('should render large size', () => {
      const { container } = render(
        <Badge
          achievement={mockAchievement}
          unlocked={true}
          size="large"
        />
      );

      const badgeElement = container.querySelector('.w-32');
      expect(badgeElement).toBeInTheDocument();
    });
  });

  describe('Tier Styling', () => {
    it('should apply bronze tier styling', () => {
      const bronzeAchievement = { ...mockAchievement, tier: 'bronze' };

      const { container } = render(
        <Badge
          achievement={bronzeAchievement}
          unlocked={true}
        />
      );

      const badgeElement = container.querySelector('.from-orange-400');
      expect(badgeElement).toBeInTheDocument();
    });

    it('should apply silver tier styling', () => {
      const silverAchievement = { ...mockAchievement, tier: 'silver' };

      const { container } = render(
        <Badge
          achievement={silverAchievement}
          unlocked={true}
        />
      );

      const badgeElement = container.querySelector('.from-gray-300');
      expect(badgeElement).toBeInTheDocument();
    });

    it('should apply gold tier styling', () => {
      const goldAchievement = { ...mockAchievement, tier: 'gold' };

      const { container } = render(
        <Badge
          achievement={goldAchievement}
          unlocked={true}
        />
      );

      const badgeElement = container.querySelector('.from-yellow-400');
      expect(badgeElement).toBeInTheDocument();
    });

    it('should apply platinum tier styling', () => {
      const platinumAchievement = { ...mockAchievement, tier: 'platinum' };

      const { container } = render(
        <Badge
          achievement={platinumAchievement}
          unlocked={true}
        />
      );

      const badgeElement = container.querySelector('.from-blue-300');
      expect(badgeElement).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should have title attribute for description', () => {
      const { container } = render(
        <Badge
          achievement={mockAchievement}
          unlocked={true}
        />
      );

      const badgeElement = container.querySelector('[title]');
      expect(badgeElement).toHaveAttribute('title', mockAchievement.description);
    });
  });

  describe('Locked State', () => {
    it('should apply opacity and grayscale when locked', () => {
      const { container } = render(
        <Badge
          achievement={mockAchievement}
          unlocked={false}
        />
      );

      const badgeElement = container.querySelector('.opacity-50.grayscale');
      expect(badgeElement).toBeInTheDocument();
    });

    it('should show description instead of points when locked', () => {
      render(
        <Badge
          achievement={mockAchievement}
          unlocked={false}
          showDetails={true}
        />
      );

      expect(screen.getByText(mockAchievement.description)).toBeInTheDocument();
      expect(screen.queryByText('+50 pts')).not.toBeInTheDocument();
    });
  });
});
