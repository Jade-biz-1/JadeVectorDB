# EnterpriseRAG User Guide

**For Field Engineers and Mechanics**

This guide will help you quickly find maintenance information using the EnterpriseRAG Q&A system.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Asking Questions](#asking-questions)
3. [Understanding Answers](#understanding-answers)
4. [Best Practices](#best-practices)
5. [Troubleshooting](#troubleshooting)
6. [FAQ](#faq)

---

## Getting Started

### Accessing the System

1. Open your web browser
2. Navigate to: **http://localhost:5173**
3. You'll see the main query interface

### System Overview

EnterpriseRAG helps you find maintenance information by:
- Searching through all indexed maintenance documentation
- Finding relevant sections across multiple manuals
- Providing answers with source citations
- Showing you exactly where information came from

---

## Asking Questions

### How to Ask Effective Questions

**✅ Good Questions:**
- Be specific: "How do I replace the hydraulic fluid in HP-3000?"
- Ask about procedures: "What is the maintenance schedule for air compressors?"
- Troubleshooting: "Why is my conveyor system overheating?"
- Safety: "What safety precautions are needed for hydraulic pump maintenance?"
- Specifications: "What type of oil does the HP-3000 use?"

**❌ Avoid Vague Questions:**
- Too broad: "Tell me about pumps"
- No context: "How to fix it?"
- Too specific to your situation: "Why is pump #4 in Building B leaking?" (the system doesn't know your specific equipment)

### Question Types the System Handles Well

| Question Type | Examples | Why It Works |
|---------------|----------|--------------|
| **How-to** | "How do I replace the air filter?"<br>"How to drain hydraulic fluid?" | System finds step-by-step procedures |
| **What** | "What is the fluid capacity?"<br>"What tools are needed?" | System retrieves specifications |
| **When** | "When should I replace filters?"<br>"When to schedule maintenance?" | System finds maintenance schedules |
| **Why** | "Why is pressure dropping?"<br>"Why does it overheat?" | System finds troubleshooting info |
| **Where** | "Where is the drain valve?"<br>"Where to find serial number?" | System locates component information |

### Using Filters

**Device Type Filter:**
- **"All Devices"**: Searches all documentation
- **"Hydraulic Pump"**: Only hydraulic pump manuals
- **"Air Compressor"**: Only air compressor docs
- **"Conveyor System"**: Only conveyor manuals

💡 **Tip:** Use device-specific filters when you know the equipment type for more focused results.

**Number of Sources (top-k):**
- **3 sources**: Quick answers, less context
- **5 sources**: ✅ **Recommended** - Good balance
- **10 sources**: More comprehensive, slower

---

## Understanding Answers

### Answer Components

When you ask a question, you'll receive:

#### 1. **The Answer**
- Generated from relevant documentation
- Written in clear, actionable language
- Often includes step-by-step instructions

Example:
```
To replace the hydraulic fluid:

1. Ensure system is cool and depressurized
2. Open drain valve at bottom of reservoir
3. Allow complete drainage (15-20 minutes)
4. Clean reservoir interior with lint-free cloth
5. Replace filter cartridge
6. Close drain valve
7. Add new hydraulic fluid (ISO VG 46)
8. Bleed air from system
```

#### 2. **Confidence Score**
- **High (>80%)**: Answer is well-supported by documentation
- **Medium (60-80%)**: Answer is relevant but may need verification
- **Low (<60%)**: Limited information found, verify independently

#### 3. **Processing Time**
- Shows how long the query took (typically 2-4 seconds)

#### 4. **Mode Badge**
- **Mock**: Demo mode with simulated responses
- **Production**: Real search through your documents

#### 5. **Source Citations**

Each answer includes sources showing where information came from:

```
Source 1: hydraulic_pump_manual.pdf
  • Pages: 23-24
  • Section: Component Replacement
  • Relevance: 95%
  • Excerpt: "...the procedure for hydraulic fluid
    replacement requires following standard safety
    protocols..."
```

**What Sources Tell You:**
- **Document name**: Which manual has the information
- **Page numbers**: Where to find it for more detail
- **Section**: What part of the manual
- **Relevance**: How closely it matches your question
- **Excerpt**: Preview of the content

### When to Trust Answers

**✅ Trust the answer when:**
- Confidence score is >80%
- Multiple sources (3+) support the answer
- Sources are from official manuals
- Information matches your training

**⚠️ Verify the answer when:**
- Confidence score is 60-80%
- Only 1-2 sources found
- Answer seems incomplete
- Safety-critical procedure

**❌ Don't trust blindly if:**
- Confidence score is <60%
- Sources are irrelevant
- Answer contradicts your knowledge
- Critical safety procedure

### Example Questions & Answers

#### Example 1: Maintenance Schedule

**Question:** "What is the monthly maintenance for hydraulic pumps?"

**Answer:**
```
Monthly maintenance for hydraulic pumps includes:

1. Replace air filters
2. Check electrical connections
3. Inspect hoses for wear and cracks
4. Test pressure relief valve
5. Clean cooling fins
6. Lubricate shaft bearings

Detailed procedures are in Section 6 (Preventive Maintenance).
```

**Sources:** hydraulic_pump_manual.pdf, Pages 42-43
**Confidence:** 92%

#### Example 2: Troubleshooting

**Question:** "My air compressor won't start. What should I check?"

**Answer:**
```
If the air compressor won't start, check:

1. Power supply and circuit breakers
2. Emergency stop button is not engaged
3. Control panel for error codes
4. Safety interlocks are not triggered
5. Thermal overload protection

If all checks pass and it still won't start, contact
a certified technician.
```

**Sources:** air_compressor_guide.pdf, Pages 67-68
**Confidence:** 88%

---

## Best Practices

### 1. Ask One Question at a Time
❌ "How do I replace the filter and what oil should I use and when should I do it?"
✅ "How do I replace the hydraulic filter?" (then ask about oil separately)

### 2. Use Specific Equipment Names
❌ "How to maintain the pump?"
✅ "How to maintain the HP-3000 hydraulic pump?"

### 3. Include Context in Your Question
❌ "Won't start"
✅ "Air compressor won't start after maintenance"

### 4. Reference Error Codes
✅ "What does error code E47 mean on the conveyor system?"

### 5. Use Technical Terms from Your Training
✅ "How to adjust the solenoid valve on hydraulic system?"

### 6. Ask Follow-up Questions
If the first answer is incomplete:
- "Can you provide more details about step 3?"
- "What tools are needed for this procedure?"
- "Are there any safety warnings for this task?"

### 7. Check Sources
Always note which manual the information came from. You can:
- Reference the physical manual for diagrams
- Show supervisors the source for verification
- Find related information in the same section

---

## Troubleshooting

### "No relevant information found"

**Cause:** Your question might be too specific or use terms not in the documentation.

**Solutions:**
1. Rephrase using simpler terms
2. Try a more general question first
3. Check if the device type filter is too restrictive
4. Verify documentation for that equipment is uploaded

**Example:**
- Instead of: "How to calibrate the differential pressure transducer?"
- Try: "How to calibrate pressure sensors?"

### Low Confidence Scores

**Cause:** Limited or unclear information in documentation.

**Solutions:**
1. Check multiple sources in the answer
2. Cross-reference with physical manuals
3. Ask a more specific question
4. Consult with supervisor or technician

### Answer Seems Wrong

**Causes:**
- Question was ambiguous
- Multiple equipment types have similar procedures
- Documentation might be outdated

**Solutions:**
1. Check the source citations - is it the right manual?
2. Verify device type filter matches your equipment
3. Ask a more specific question with model number
4. Report to your supervisor

### System is Slow

**Normal:** 2-4 seconds is typical
**Slow:** >10 seconds

**Solutions:**
1. Reduce number of sources (try 3 instead of 10)
2. Check your internet connection (if using remote server)
3. Wait for ongoing document processing to complete
4. Report to system administrator

---

## FAQ

### General Questions

**Q: Can I ask questions in my own words?**
A: Yes! The system understands natural language. Ask as if you're talking to an experienced technician.

**Q: Does the system have information about my specific equipment?**
A: Only if the maintenance manual has been uploaded. Check the Admin Panel to see which documents are indexed.

**Q: Can I trust the answers for safety-critical procedures?**
A: Always verify safety-critical procedures with:
- Official manuals (use the page numbers provided)
- Your supervisor
- Certified technician
Never skip safety steps based solely on system answers.

**Q: What if I don't find what I need?**
A:
1. Try rephrasing your question
2. Check if the manual exists in the system
3. Consult physical manuals
4. Ask your supervisor
5. Request that missing documentation be added

### Using the System

**Q: Can I see my previous questions?**
A: Not currently. Write down important answers or page references.

**Q: Can I save or print answers?**
A: Yes! Use your browser's print function (Ctrl+P or Cmd+P) or copy/paste the answer.

**Q: Can I search for part numbers?**
A: Yes, if part numbers are mentioned in the documentation. Example: "Where can I find part #HF-3000-F?"

**Q: What languages does it support?**
A: Currently English only.

**Q: Can I upload my own documents?**
A: No, only administrators can upload documents. Request additions through your supervisor.

### Interpreting Results

**Q: What does "relevance" percentage mean?**
A: How closely the source matches your question:
- 90-100%: Highly relevant
- 80-89%: Relevant
- 70-79%: Somewhat relevant
- <70%: May not be directly related

**Q: Why do I get different answers for the same question?**
A: If you change filters (device type, number of sources), you might get different answers as the system searches different document sets.

**Q: What if two sources contradict each other?**
A: This can happen if:
- Documents are from different equipment models
- One document is outdated
- Different manufacturers have different procedures
Always check source citations and verify with the most recent official manual.

**Q: Can I provide feedback on answers?**
A: Yes! Note incorrect answers and report to your supervisor so documentation can be improved.

---

## Tips for Common Scenarios

### Before Starting Work
1. Ask: "What safety precautions are needed for [equipment] maintenance?"
2. Ask: "What tools and parts are needed for [procedure]?"
3. Review the complete procedure before starting

### During Maintenance
1. If stuck, ask specific questions about the current step
2. Check torque specifications, fluid types, etc.
3. Look up error codes if they appear

### Troubleshooting
1. Start with: "What are common problems with [equipment]?"
2. Then ask about specific symptoms
3. Follow diagnostic procedures step-by-step

### After Maintenance
1. Verify: "What should I check after [procedure]?"
2. Document any issues found
3. Report missing or unclear documentation

---

## Getting Help

### System Issues
- **Cannot access system**: Contact IT support
- **Error messages**: Take a screenshot and report to admin
- **Slow performance**: Report to system administrator

### Content Issues
- **Missing documentation**: Request upload through supervisor
- **Incorrect information**: Report to documentation team
- **Outdated manuals**: Request update through proper channels

### Training
- **Not sure how to use system**: Review this guide
- **Need hands-on training**: Request session with supervisor
- **Have suggestions**: Provide feedback to documentation team

---

## Quick Reference Card

**Print this section and keep it at your workstation!**

### Basic Steps
1. Open http://localhost:5173
2. Type your question
3. Select device type (if known)
4. Click "Ask Question"
5. Review answer and sources
6. Note page numbers for reference

### Question Templates
- "How do I [task] on [equipment]?"
- "What is the [specification] for [equipment]?"
- "Why is [equipment] [symptom]?"
- "When should I [maintenance task]?"
- "What does error code [code] mean?"

### Quality Checks
✅ Confidence >80%
✅ Multiple sources
✅ Recent documentation
✅ Matches your training

⚠️ Always verify safety-critical procedures
⚠️ Cross-check with physical manuals when needed
⚠️ Report incorrect information

### Support
- **System Help**: [Your IT Support]
- **Documentation**: [Your Documentation Team]
- **Emergency**: Follow standard procedures, don't rely on system alone

---

**Version 1.0** | Last Updated: March 28, 2026
**Questions?** Contact your supervisor or system administrator
