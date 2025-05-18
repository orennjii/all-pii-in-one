**Python 项目文件结构组织指南 (供参考)**

## 1. 核心原则

  - **模块化 (Modularity)**: 每个模块（文件或目录）应该专注于一个明确定义的功能或责任。避免创建过于庞大、功能混杂的模块。
  - **高内聚 (High Cohesion)**: 将相关性高的代码（例如，处理同一类数据或实现同一项功能的类和函数）组织在一起。
  - **低耦合 (Low Coupling)**: 模块之间应尽可能减少依赖，使得修改一个模块不会轻易影响到其他模块。
  - **可读性 (Readability)**: 清晰、一致的结构使得其他开发者（以及未来的你）更容易理解项目的组织方式和代码逻辑。
  - **可维护性 (Maintainability)**: 良好的结构便于定位和修改代码，降低维护成本。
  - **可扩展性 (Scalability/Extensibility)**: 当项目需求增加时，清晰的结构更容易添加新功能或扩展现有功能，而不会破坏原有结构。
  - **可测试性 (Testability)**: 合理的结构使得单元测试和集成测试的编写更加容易。

## 2. 代码风格

  - **遵循 Google Python Style Guide**: 这主要体现在代码的编写规范上，例如命名约定、类型注解、代码格式、DocString等。
    - [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) 
    - 以下是一些 google 风格指南的示例::
      - 代码行长度限制为 80 个字符
      - 使用四个空格进行缩进
      - 函数和变量命名使用小写字母和下划线分隔
      - 类名使用大写字母开头的驼峰命名法
  - **行内注释**: 代码和注释**绝对禁止**在一行中混合, 如果需要用到行注释, 必须是先注释, 然后是代码。
  - **模块导入**: 导入语句应该放在文件的顶部，并按照System modules、External modules和Internal modules的顺序排列, 这三种不同模块之间应该有空行分隔。对于内部模块, 如果是同一目录下的模块, 则使用相对导入, 否则使用绝对导入。
  - **括号使用 (Parentheses Usage)**: 对于使用圆括号的语句, 例如:
    ```python
    # 正确
    example(args1, args2, args3):

    example(
        args1, args2, args3
    )

    example(
        args1,
        args2,
        args3,
    )
        
    # 错误
    example(function_one(arg1, arg2)
              + function_two(arg3, arg4))
    ```
  - **标点符号使用 (Punctuation Usage)**: 在注释中禁止使用中文标点符号, 而是使用对应的英文半角字符