# Sample Data

This directory contains sample maintenance documentation for testing EnterpriseRAG.

## Sample Documents

### For Testing (Production Mode)

When running in production mode, you can upload actual PDF or DOCX files here for testing.

**Example maintenance manuals to test with:**
- Hydraulic pump maintenance manual
- Air compressor service guide
- Conveyor system documentation
- Industrial equipment troubleshooting guide

## Mock Mode

In mock mode, the system uses pre-programmed responses and doesn't require actual documents. This is perfect for:
- Demonstrations
- UI testing
- Development
- Evaluating the interface before setting up production components

## Creating Test Documents

If you want to create simple test documents:

### Sample Maintenance Manual Template

```
HYDRAULIC PUMP MAINTENANCE MANUAL
Model: HP-3000
==================================

Section 1: SAFETY GUIDELINES
-----------------------------
1. Always disconnect power before maintenance
2. Wear appropriate PPE (gloves, safety glasses)
3. Release hydraulic pressure before servicing
4. Follow lockout/tagout procedures

Section 2: SPECIFICATIONS
-------------------------
- Flow Rate: 50 GPM
- Maximum Pressure: 3000 PSI
- Hydraulic Fluid: ISO VG 46
- Fluid Capacity: 15 liters
- Operating Temperature: 10-50°C

Section 3: ROUTINE MAINTENANCE
-------------------------------
Daily:
- Visual inspection for leaks
- Check fluid level
- Verify pressure readings

Weekly:
- Inspect filters
- Check belt tension
- Lubricate moving parts

Monthly:
- Replace air filters
- Check electrical connections
- Inspect hoses for wear

Quarterly:
- Change hydraulic fluid
- Inspect safety systems
- Calibrate pressure sensors

Annually:
- Complete system overhaul
- Replace wear parts
- Professional inspection

Section 4: FLUID REPLACEMENT PROCEDURE
---------------------------------------
1. Ensure system is cool and depressurized
2. Open drain valve at bottom of reservoir
3. Allow complete drainage (15-20 minutes)
4. Clean reservoir interior with lint-free cloth
5. Replace filter cartridge (Part #HF-3000-F)
6. Close drain valve
7. Add new hydraulic fluid (ISO VG 46)
8. Run system at low pressure to bleed air
9. Check fluid level and top up to FULL mark

Section 5: TROUBLESHOOTING
---------------------------
Problem: Pump won't start
- Check power supply and circuit breakers
- Verify emergency stop is not engaged
- Check for error codes on control panel
- Inspect safety interlocks

Problem: Low pressure
- Check fluid level
- Inspect for leaks
- Replace clogged filters
- Check pump relief valve setting

Problem: Overheating
- Check cooling fan operation
- Clean air vents and heat exchanger
- Verify coolant circulation
- Reduce operating load
- Check ambient temperature

Problem: Unusual noise
- Check for cavitation (low fluid)
- Inspect for air in system
- Check coupling alignment
- Listen for bearing wear
```

Save this as a `.txt` file, convert to PDF, and upload via the admin panel!

## Tips for Testing

1. **Start with mock mode** to understand the interface
2. **Use simple test documents** before uploading large manuals
3. **Test different device types** (hydraulic_pump, air_compressor, etc.)
4. **Ask various question types**:
   - How-to questions
   - Troubleshooting
   - Maintenance schedules
   - Safety procedures

## Production Documents

For production use, prepare your actual maintenance documentation:
- Equipment manuals
- Service guides
- Troubleshooting procedures
- Safety documentation
- Parts catalogs

Supported formats: **PDF**, **DOCX**
