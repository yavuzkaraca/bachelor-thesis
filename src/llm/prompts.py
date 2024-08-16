"""
This module provides template functions that generate standard prompts and guidelines for validating Software
Specification Documents.

Functions:
    - instructions: Returns a detailed prompt instructing an assistant on how to identify and report incompleteness in
    Software Specification Documents.
    - ieee_guidelines: Returns a description of the importance of completeness in Software Requirements Specifications
    (SRS) and outlines IEEE guidelines for ensuring a complete SRS document.
    - completeness_types: Returns a detailed explanation of the concept of completeness in Software Requirements
    Specifications (SRS) and outlines three levels of completeness in SRS documents.

"""


def system_engineer_role():
    return """
    You are an experienced Requirements Engineer that identifies incompleteness in Software Requirement Specification 
    documents.
    """


def system_default_role():
    return """
    You are an AI assistant that validates completeness in Software Requirement Specification documents.
    """


def instructions_few_shot():
    """
    Provides a prompt for an assistant tasked with identifying incompleteness in Software Specification Documents.

    The prompt details the steps the assistant should follow:
        1. Refer to the requirement by its unique identifier, section number, or "N/A" if it's document-wide.
        2. State the issue found.
        3. Offer a correction or suggestion.

    Output Format:
        The output is expected to be a CSV file with columns: "Label", "Issue", and "Suggestion".

    Returns:
        A string containing the instructions and an example output.
    """
    return """
    Identify all the instances of incompleteness in the following document by:
    1. Referring to the requirement by its unique identifier/label. If it's more than one requirement then you can
     refer to the section number. If the section is totally missing than you can say "N/A".
    2. Stating the issue.
    3. Providing an example of a requirement that addresses the identified incompleteness.

    Output Format:
    Produce a CSV file with the following columns: "Label", "Issue", and "Correction". Ensure that each cell 
    value in the CSV file is separated with a semicolon (;).

    Example Output:
    R19;does not specify what happens if the message is ignored;If the immunization reminder is ignored, the system shall send an alert to the administrator
    Section 5.1;does not specify total supported number of concurrent users;The system shall handle up to 10.000 concurrent users without performance degradation
    FR-A-30;does not specify recovery or program state when selection is improper;Improper Selection is undone and the previous state before the selection is still valid
    \n
    """


def instructions_chain_of_thought():
    """
    Provides a prompt for an assistant tasked with identifying incompleteness in Software Specification Documents.

    The prompt details the steps the assistant should follow:
        1. Refer to the requirement by its unique identifier, section number, or "N/A" if it's document-wide.
        2. State the issue found.
        3. Offer a correction or suggestion.

    Output Format:
        The output is expected to be a CSV file with columns: "Label", "Issue", and "Suggestion".

    Returns:
        A string containing the instructions and an example output.
    """
    return """
    First, read the following document thoroughly and identify all the instances of incompleteness.
    Then, take your time and think about these instances if they really are accurate and valid.
    After that, explain the reason of incompleteness for each instance referring to their unique identifier.
    Finally, depending on the reason, provide an example of a requirement that addresses the identified incompleteness.

    Output Format:
    Produce a CSV file with the following columns: "Label", "Issue", and "Correction". Ensure that each cell 
    value in the CSV file is separated with a semicolon (;).
    \n
    """


def instructions_zero_shot():
    """
    Provides a prompt for an assistant tasked with identifying incompleteness in Software Specification Documents.

    The prompt details the steps the assistant should follow:
        1. Refer to the requirement by its unique identifier, section number, or "N/A" if it's document-wide.
        2. State the issue found.
        3. Offer a correction or suggestion.

    Output Format:
        The output is expected to be a CSV file with columns: "Label", "Issue", and "Suggestion".

    Returns:
        A string containing the instructions and an example output.
    """
    return """
    Identify all the instances of incompleteness in the following document by:
    1. Referring to the requirement by its unique identifier/label. If it's more than one requirement then you can
     refer to the section number. If the section is totally missing than you can say "N/A".
    2. Stating the issue.
    3. Providing an example of a requirement that addresses the identified incompleteness.

    Output Format:
    Produce a CSV file with the following columns: "Label", "Issue", and "Correction". Ensure that each cell 
    value in the CSV file is separated with a semicolon (;).
    \n
    """


def generate_ieee_guidelines():
    return """
    Provide the IEEE guidelines for Software Requirement Specification documents about the Completeness aspect. 
    According to IEEE, documents should fulfill three things to be complete. What are these?
    """


def completeness_types():
    """
    Provides a detailed explanation of the concept of completeness in Software Requirements Specifications (SRS)
    and the importance of adhering to IEEE guidelines. The function outlines three levels of completeness in SRS
    documents, emphasizing the implications of incompleteness and the necessity for comprehensive requirement
    documentation.
    """
    return """
    7 Completeness

    The thesis defines three levels of completeness to assess the capabilities of 
    LLMs in identifying different types of completeness:

    1. **Simple Completeness**:
       - Involves identifying easily detectable gaps, such as missing elements in
         lists, labels, and references, without needing contextual understanding.

    2. **Explicit Semantic Completeness**:
       - Focuses on identifying gaps in essential requirements like functionality,
         performance, and design constraints by evaluating the provided 
         information in the document.

    3. **Implicit Domain Knowledge Requiring Completeness**:
       - Pertains to identifying assumed but undocumented domain-specific knowledge
         that requires external understanding beyond what is explicitly stated 
         in the document.

    The categorization is informed by foundational research in text comprehension, 
    which emphasizes coherence and the cognitive processes involved in understanding
    text [25]. Just as these coherence types help readers understand and connect
    textual elements, the levels of completeness facilitate a structured approach 
    to identifying gaps, ranging from the obvious to those requiring domain knowledge.

    ### 7.1 Simple Completeness

    Incompleteness at this level encompasses easily identifiable gaps, such as missing
    elements in lists, and formal or reference incompleteness [16]. This corresponds
    to the violation of the IEEE guideline to provide "complete labels and references
    for all figures, tables, and diagrams, and define all terms and units of measure" 
    [13].

    The key aspect of validating simple completeness is that it does not necessitate an
    understanding of the project or its context, thereby eliminating the need for deeper
    comprehension or interpretation. Essentially, verifying simple completeness is 
    comparable to conducting a syntax analysis.

    #### 7.1.1 Formal Completeness

    Formal Completeness, as delineated by Kuchta [16], encompasses:

    1. **Template Completeness**:
       - Refers to the extent to which all required sections and elements within a
         document template are present and accounted for. Measuring template 
         completeness involves counting the elements required by the document template, 
         also known as "the number of required metaclasses" [16], and identifying any 
         missing elements. For instance, if a document lacks an entire chapter, such as
         an introduction or constraints, this results in template incompleteness. This 
         thesis extends the notion of template completeness beyond document-wide 
         templates to include specific requirement-driven templates. If a specific 
         requirement establishes a template, such as "the program does the following
         three modes," then the items of this list can be validated similarly, and a 
         missing item would result in a failure to fulfill the template.

    2. **Definition Completeness**:
       - Assesses the completeness of definitions within the document. It examines 
         whether all required properties of elements are fully defined, identifying any 
         elements with incomplete definitions.

    #### 7.1.2 Reference Completeness

    Kuchta [16] also defines Reference Completeness, which includes:

    1. **Solution Completeness**:
       - Evaluates the completeness of the "trace" references that lead to solutions 
         within the SRS. It includes specific parts of the SRS that explain the design,
         architecture, or implementation details necessary to achieve the requirements.
         These descriptions can cover algorithms, modules, subsystems, interfaces, data
         structures, and any other technical details that outline how the system will 
         function and meet its specified requirements.

    2. **Internal Reference**:
       - Evaluates whether all required internal references (e.g., cross-references 
         within the document) are present and correctly linked. This ensures that the 
         document is self-contained and that references within it lead to the correct 
         sections or elements, facilitating ease of navigation and understanding.

    ### 7.2 Explicit Semantic Completeness

    Incompleteness at this level manifests as gaps in essential requirements, such as
    functionality, performance, and design constraints, violating the IEEE guideline 
    that a document should contain "all significant requirements, whether relating to 
    functionality, performance, design constraints, attributes, or external interfaces" 
    [13]. Additionally, this level involves interactions or dependencies between 
    requirements that are not clearly defined or where the behavior is only partially 
    specified or completely unspecified. These gaps are identifiable solely based on the 
    information provided in the document.

    #### 7.2.1 Functional Completeness

    Functional Completeness evaluates whether all system functionalities and behaviors 
    are thoroughly and clearly defined, adhering to the IEEE guideline that a document 
    should include "definition of the responses of the software to all realizable 
    classes of input data in all realizable classes of situations" [13].

    By examining the functional requirements in the dataset, common examples of 
    incompleteness were identified, including but not limited to:

    1. **Under-specified Requirements**:
       - Some requirements may be described only at a high level without sufficient 
         detail. For example, "The system shall allow the user to upload files" may not 
         specify file size limits, file types, or how to handle unsupported file types.

    2. **Dependent Requirements**:
       - Certain requirements may rely on other functionalities that are not specified,
         leading to incomplete implementations. For instance, "The system shall confirm 
         successful file uploads" presumes an existing functionality that enables file 
         uploads.

    3. **Partial Behavior Specifications**:
       - Often, only portions of a behavior are described, leaving out crucial states 
         and actions of a system. For example, a "pause" function might be detailed 
         without any requirements describing a "resume" function, or a multi-step 
         approval process might lack guidelines on handling rejections and recovery 
         procedures.

    4. **Undefined Inputs**:
       - The system might not specify behavior for all possible inputs, whether 
         numerical, textual, or user interactions like clicks, especially under unusual 
         conditions. For example, the requirement that "User should be able to set the 
         speed of the simulation by entering a number" fails to define the behavior when 
         zero or negative numbers are entered. Similarly, how the system responds to 
         invalid inputs may not be specified. The IEEE guidelines on completeness 
         emphasize this: "Note that it is important to specify the responses to both 
         valid and invalid input values" [13].

    #### 7.2.2 Non-functional Completeness

    Non-functional requirements are crucial for defining the quality and operational 
    standards of a system, encompassing performance, usability, reliability, security, 
    maintainability, and scalability [2]. Non-functional incompleteness arises when 
    these critical aspects are not specified in detail. For example, a system might lack 
    clear specifications on performance factors such as speed, response time, and 
    throughput, or it may omit crucial details regarding user interface design and user 
    experience enhancements. Similarly, specifications might not address reliability 
    concerns like uptime and recovery procedures, or they may fail to outline necessary 
    security measures including data protection, access control, and threat mitigation. 
    Furthermore, the absence of guidelines on code maintainability, documentation, and 
    how the system should scale with increased loads also contributes to non-functional 
    incompleteness.

    #### 7.2.3 Constraint Completeness

    Incompleteness in constraints could result in missing elements in critical areas:

    1. **Interface Requirements**:
       - Includes missing details on how different system modules will interact, 
         insufficient specifications for interactions with external systems, APIs, and 
         third-party services, as well as gaps in user interface design, including 
         incomplete information on screens, inputs, and outputs.

    2. **Environmental Requirements**:
       - There are gaps in specifications concerning the hardware required to run the 
         software. Additionally, details on necessary operating systems, middleware, and 
         other essential software dependencies are often lacking.

    3. **Data Requirements**:
       - There is an absence of necessary entities, attributes, or relationships, which 
         are crucial for the system’s data integrity and functionality.

    ### 7.3 Implicit Domain Knowledge Requiring Completeness

    This level pertains to the assumed knowledge about the application domain that may 
    not be explicitly stated in the SRS document [10, 21]. Incompleteness at this level 
    cannot be identified from the document alone, as it requires external knowledge 
    about the application domain. Often, this type of incompleteness arises from 
    overlooked or implicitly assumed functionalities that are vital to the application 
    but not formally documented.

    For example, in the development of a football video game, developers might 
    implicitly assume that the game should include mechanisms to resolve tie games 
    according to football rules, such as extra time and penalty shootouts. If these 
    gameplay elements are not explicitly specified in the SRS, the game may lack these 
    features and adherence to real-world football dynamics that players expect.

    Similarly, a trading platform might operate under the implicit assumption that it 
    will execute trades within milliseconds to capitalize on fast-moving markets. If the 
    need for low latency is not explicitly captured in the requirements, the final 
    product might fail to meet the critical operational standards of the trading sector. 
    Such an oversight can impact the platform’s competitiveness and ability to function 
    effectively in a high-stakes environment where every millisecond counts.
    \n
    """