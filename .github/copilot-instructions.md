<copilot_instructions>
  <core_principles>
    - output full functions and classes in full fidelity
    - optimize for readability and maintainability over cleverness
    - use descriptive variable, function, class, parameter, and method names
    - keep functions focused on a single responsibility
    - prefer explicit error handling over silent failures
    - minimize external dependencies when practical
    - summarize your changes and potential optimizations at the bottom of each discussion
  </core_principles>

  <code_style>
    - use lowercase letters at the beginning of a sentence in comments
    - use comments sparingly and only where necessary
    - comment at the function level using swagger/openapi standards in the respective language
    - prefer code that explains itself over explanatory comments
    - use TODO/FIXME/NOTE prefixes for actionable comments with context
    - follow consistent indentation and formatting standards throughout the code
    - keep line length reasonable (80-120 characters)
    - use language idioms appropriately for clarity and performance
  </code_style>

  <structure_and_organization>
    - prefer composition over inheritance
    - avoid deep nesting (max 3-4 levels)
    - extract complex logic into well-named helper functions
    - separate business logic from framework/infrastructure code
    - group related imports and separate from standard library imports
    - organize files and modules by feature/domain rather than type
    - use consistent file naming conventions
  </structure_and_organization>

  <error_handling>
    - use specific exception types rather than generic ones
    - validate inputs at function boundaries
    - fail fast and fail clearly with meaningful error messages
    - handle edge cases explicitly
    - validate and sanitize all external inputs
    - use secure defaults and fail securely
  </error_handling>

  <dependencies_and_architecture>
    - prefer standard library solutions when available
    - isolate external dependencies behind interfaces
    - use dependency injection for testability
    - externalize configuration from code
    - provide sensible defaults for optional configurations
  </dependencies_and_architecture>

  <documentation_and_types>
    - use type hints and docstrings for better code clarity
    - include examples in docstrings for complex functions
    - document invariants, preconditions, and postconditions
    - keep documentation close to code and update together
    - document required environment variables and setup steps
    - when creating markdown files, use the CommonMark format
  </documentation_and_types>

  <performance_and_security>
    - avoid premature optimization but consider algorithmic complexity
    - use appropriate data structures for the use case
    - avoid hardcoded secrets or sensitive data
    - log structured data when possible
    - include logging at appropriate levels (debug, info, warn, error)
  </performance_and_security>

  <testing_and_quality>
    - write unit tests for business logic and complex functions
    - use descriptive test names that explain the scenario
    - prefer integration tests for critical user flows
    - make atomic commits that represent single logical changes
    - avoid committing commented-out code or debug statements
    - follow established conventions for the target language/framework
  </testing_and_quality>

  <output_requirements>
    - provide reasoning for architectural decisions when significant
    - highlight any breaking changes or migration considerations
    - include relevant context for complex implementations
    - stay within supported language versions unless compelling reason
  </output_requirements>
</copilot_instructions>
