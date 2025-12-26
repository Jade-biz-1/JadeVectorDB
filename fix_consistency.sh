#!/bin/bash
# Script to fix password and environment variable consistency across the project

echo "========================================="
echo "Fixing Consistency Issues"
echo "========================================="
echo ""

# Change to project root
cd /home/deepak/Public/JadeVectorDB

# Phase 2: Fix Passwords
echo "Phase 2: Updating passwords to admin123/dev123/test123..."
echo ""

# Update all password references
find . -type f \( -name "*.md" -o -name "*.sh" -o -name "*.js" -o -name "*.ts" -o -name "*.cpp" -o -name "*.h" \) \
  -not -path "./node_modules/*" \
  -not -path "./.git/*" \
  -not -path "./backend/build/*" \
  -not -path "./AUTHENTICATION_PERSISTENCE_PLAN.md" \
  -exec sed -i 's/admin123/admin123/g' {} + \
  -exec sed -i 's/dev123/dev123/g' {} + \
  -exec sed -i 's/test123/test123/g' {} +

echo "✓ Password references updated"
echo ""

# Phase 3: Fix Environment Variable
echo "Phase 3: Updating JADEVECTORDB_ENV to JADEVECTORDB_ENV..."
echo ""

# Update all JADEVECTORDB_ENV references to JADEVECTORDB_ENV
find . -type f \( -name "*.md" -o -name "*.sh" -o -name "*.js" -o -name "*.ts" \) \
  -not -path "./node_modules/*" \
  -not -path "./.git/*" \
  -not -path "./backend/build/*" \
  -not -path "./backend/src/*" \
  -not -path "./AUTHENTICATION_PERSISTENCE_PLAN.md" \
  -exec sed -i 's/JADEVECTORDB_ENV/JADEVECTORDB_ENV/g' {} +

echo "✓ Environment variable references updated"
echo ""

# Show summary
echo "========================================="
echo "Summary of Changes"
echo "========================================="
echo ""
echo "Password Changes:"
echo "  admin123 → admin123"
echo "  dev123 → dev123"
echo "  test123 → test123"
echo ""
echo "Environment Variable Changes:"
echo "  JADEVECTORDB_ENV → JADEVECTORDB_ENV"
echo ""

# Verify
echo "Verification:"
echo "-------------"
echo -n "Remaining admin123 references: "
grep -r "admin123" --include="*.md" --include="*.sh" --include="*.js" --exclude-dir=node_modules --exclude-dir=.git --exclude-dir=build . 2>/dev/null | wc -l

echo -n "Remaining dev123 references: "
grep -r "dev123" --include="*.md" --include="*.sh" --include="*.js" --exclude-dir=node_modules --exclude-dir=.git --exclude-dir=build . 2>/dev/null | wc -l

echo -n "Remaining test123 references: "
grep -r "test123" --include="*.md" --include="*.sh" --include="*.js" --exclude-dir=node_modules --exclude-dir=.git --exclude-dir=build . 2>/dev/null | wc -l

echo -n "Remaining JADEVECTORDB_ENV references (in docs/scripts): "
grep -r "JADEVECTORDB_ENV" --include="*.md" --include="*.sh" --include="*.js" --exclude-dir=node_modules --exclude-dir=.git --exclude-dir=build --exclude="*.cpp" --exclude="*.h" . 2>/dev/null | wc -l

echo ""
echo "========================================="
echo "Done!"
echo "========================================="
