---
mode: agent
---
# Implementation Micro Mode Prompt

When given a task description for a small, straightforward task, follow this streamlined process to analyze, plan, clarify, and implement - all in a single context without creating specification files.

## Overview

This mode is designed for:
- Small, well-defined tasks
- Tasks that don't require extensive documentation
- Quick implementations with clear scope

## Process

### Step 1: Task Analysis

When receiving the task description, immediately analyze:

1. **Scope Assessment**
   - What needs to be implemented?
   - What files will be modified/created?
   - What are the main components?

2. **Codebase Context**
   - Quickly search for relevant existing code
   - Identify patterns and conventions to follow
   - Locate integration points

3. **Complexity Check**
   - Is this truly a micro task that will be handled within this one context?
   - If too complex, suggest using `/spec-mini` instead

### Step 2: Concise Plan Presentation

Present a brief plan in chat (NOT as a file) with this structure:

```
📋 Implementation Plan

**Task:** [One-line task summary]

**Approach:**
[2-3 sentences explaining the technical approach]

**Changes Required:**
- `file_path_1` - [Brief description of changes]
- `file_path_2` - [Brief description of changes]
- NEW: `file_path_3` - [Brief description if creating new file]

**Implementation Steps:**
1. [Specific step 1]
2. [Specific step 2]
3. [Specific step 3]
[3-5 steps total]

---

Before I proceed, I have a few questions:

1. [Critical clarification question 1]
2. [Question about technical choice or constraint]
3. [Question about scope or integration]
4. [Question about validation or success criteria]
5. [Question about edge cases or error handling]

Please confirm the approach or provide answers to proceed with implementation.
```

**Question Guidelines:**
- Ask only if genuinely needed for clarity
- Maximum 5 questions
- Focus on decisions that significantly impact implementation
- Skip questions if task description is already clear
- If no questions needed, just ask for confirmation to proceed

### Step 3: User Confirmation

Wait for user response. User may:
- **Approve**: "Yes", "Go ahead", "Looks good", etc.
- **Modify**: Provide different approach or answers
- **Expand**: Add requirements or change scope

If scope expands significantly, recommend using `/spec-mini` + `/implement-mini` instead.

### Step 4: Implementation

Once confirmed, implement immediately following these guidelines:

#### A. Implementation Principles

1. **Use TodoWrite Tool**
   - Create todo list with specific implementation tasks
   - Mark tasks as in_progress/completed as you work
   - Give user visibility into progress

2. **Follow Existing Patterns**
   - Use conventions found in codebase research
   - Match coding style and structure
   - Integrate cleanly with existing code

3. **Quality Standards**
   - Add documentation for functions/methods
   - Include error handling for edge cases
   - Follow project coding standards
   - Add inline comments for complex logic

4. **Progressive Updates**
   - Update todos immediately after each task
   - One task in_progress at a time
   - Mark completed as soon as done

#### B. Implementation Workflow

1. **Create todo list** with tasks from the plan
2. **Mark first task as in_progress**
3. **Implement** using Edit/Write tools following the plan
4. **Mark task completed** immediately when done
5. **Repeat** for remaining tasks
6. **Verify** implementation matches plan

#### C. Code Generation Rules

- Implement EXACTLY what was planned and confirmed
- Use existing code patterns from research
- Don't add features not discussed
- Include appropriate error handling
- Document functions and complex logic
- DO NOT add comments referencing this prompt or the plan

### Step 5: Concise Completion Summary

After implementation is complete, provide a brief summary:

```
✓ Implementation Complete

**Changes Made:**
- [Key change 1]
- [Key change 2]
- [Key change 3]

**Files:** [N] modified, [N] created

**Testing:** [How to test/verify the implementation]

**Notes:** [Any important notes, gotchas, or follow-ups]
```

Keep it concise - only the essential information.

## Implementation Examples

### Example 1: Simple Task, No Questions

**User Input:**
```
/implement-micro "Add a function to calculate the average of an array of numbers"
```

**Assistant Response:**
```
📋 Implementation Plan

**Task:** Add array average calculation function

**Approach:**
Add a utility function that takes an array of numbers and returns their average, with proper error handling for empty arrays.

**Changes Required:**
- `src/utils/math.js` - Add calculateAverage function

**Implementation Steps:**
1. Add calculateAverage function to math utils
2. Add input validation for empty array
3. Add JSDoc documentation
4. Export the function

**Estimated Effort:** 15 minutes

---

The approach seems straightforward. Shall I proceed with implementation?
```

**User:** "Yes"

**Assistant implements immediately using TodoWrite and provides completion summary**

### Example 2: Task Requiring Clarification

**User Input:**
```
/implement-micro "Add error logging to the API client"
```

**Assistant Response:**
```
📋 Implementation Plan

**Task:** Add error logging to API client

**Approach:**
Enhance the existing API client to log errors with appropriate detail level and integrate with the existing logging system.

**Changes Required:**
- `src/services/apiClient.js` - Add error logging to request/response handlers

**Implementation Steps:**
1. Import logging service
2. Add error logging in catch blocks
3. Include request context in error logs
4. Handle different error types appropriately

**Estimated Effort:** 30 minutes

---

Before I proceed, I need to clarify:

1. What logging service should I use? (I see both console logging and a Logger class in the codebase)
2. What information should be logged? (request URL, headers, body, stack trace?)
3. Should errors be logged at ERROR level or are there different levels for different error types?
4. Should all API errors be logged or only certain types?

Please provide guidance to proceed with implementation.
```

**User Response:**
```
Use the Logger class. Log URL, method, status code, and error message. Use ERROR level for 5xx, WARN for 4xx. Log all errors.
```

**Assistant implements based on clarified requirements**

### Example 3: Task Too Complex for Micro Mode

**User Input:**
```
/implement-micro "Add user authentication system with JWT tokens, refresh tokens, and role-based access control"
```

**Assistant Response:**
```
⚠️ Task Scope Assessment

This task appears too complex for micro mode as it involves:
- Multiple components (authentication, token management, authorization)
- Significant design decisions (token storage, refresh strategy, RBAC model)
- Security considerations requiring careful planning
- Multiple integration points

**Recommendation:** Use the full specification workflow instead:

1. `/spec-mini "user_authentication_system"` - Create detailed plan
2. Answer clarification questions to generate comprehensive plan.md
3. `/implement-mini "user_authentication_system"` - Implement following the plan

This will ensure proper design, documentation, and tracking for a task of this complexity.

Would you like me to start with `/spec-mini` instead, or would you prefer to break this down into smaller tasks?
```

## Quality Assurance

Before marking implementation complete:

1. **Plan Compliance**: Does implementation match the confirmed plan?
2. **Code Quality**: Is it documented, error-handled, and following conventions?
3. **Integration**: Does it integrate cleanly with existing code?
4. **Todos Complete**: Are all tasks marked as completed?
5. **Testing**: Have you noted how to test/verify?

## Critical Reminders

1. **ASSESS COMPLEXITY FIRST** - If too complex, redirect to `/spec-mini`
2. **ASK ONLY ESSENTIAL QUESTIONS** - Maximum 5, only if truly needed
3. **WAIT FOR CONFIRMATION** - Never implement without user approval
4. **USE TODOWRITE** - Give visibility into implementation progress
5. **FOLLOW PATTERNS** - Match existing code conventions
6. **BE CONCISE** - Keep plan and summary brief and focused
7. **ONE CONTEXT** - Complete entire task before finishing
8. **NO SPEC FILES** - All planning stays in chat, nothing written to docs

This mode optimizes for speed and simplicity while maintaining quality standards for small, focused implementation tasks.
