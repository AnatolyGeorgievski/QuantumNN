/*! \file qnn_query -- разбор и заполнение шаблонов запроса в HTTP GET

    Шаблоны выделяются оп тегу начала и конца: `${`, `}`. 
    НАДО реализовать поддержку шаблонов совместимых с разными LLM и шаблонизаторами

    `{{`, `}}` -- jinja
    `<?`, `?>` -- php
    `<`, `>` -- xml

    Принципы токенизации: 
    + Токен начинается с символа `{` и заканчивается символом `}`
    + Специальные (управляющие токены) начинаются с символа `<|` и заканчиваются символом `|>`, состоят из букв и цифр, начинаются с буквы. Может содержать символ `_`.
    + Токен "```" -- многострочный токен, заканчивается токеном "```"
    + Токен `<|`имя`|>` -- токен с именем, заменяется по таблице контрольных токенов. 
    + Токен `<` имя `>` -- токен с именем, заменяется по таблице специальных токенов. 
    + Токен `<__image__>` -- токен с именем, заменяется по таблице изображений.
    Специальные слова, которые в модели воспринимаются как токены, например роли SYSTEM:, ASSISTANT: и USER: могут быть представлены в виде токенов:
     `<|system|>`, `<|assistant|>`, `<|user|>, `<|context|>`.

В модели Granite свои особенности разметки ролей и сообщений:
```
    <|start_of_role|>system<|end_of_role|>
    You are Granite, an AI language model developed by IBM in 2024.  You are a cautious assistant.  You carefully follow instructions.  
    <|end_of_text|>
```
Сокращенно, чтобы модель в точности соблюдала характер, достаточно сказать две фразы
"You are a cautious assistant.  You carefully follow instructions.  "


    простая реализация выполняет поиск и замену токенов по таблице. Для переносимости мы предлагаем использовать специальные токены ролей.

    Например, `<|tool_call|>` токен начала списка вызовов инструментов. За ним следуют токены вызова инструментов в нотации JSON:
    `[{"name": "tool_name", "arguments": {"arg1": "value1", "arg2": "value2"}, "id": "tool_call_id"}]`.

    Роли сообщений:
    *   `system`:  <begin>system и <end> - Определяет системный промпт, содержащий общие инструкции.
    *   `user`: <begin>user и <end> - Определяет ввод пользователя.
    *   `assistant`: <begin>assistant и <end> - Определяет мой ответ.
    *   `observer`: <begin>observer и <end> - Определяет сообщение от наблюдателя.
    *   `tool`: <begin>tool и <end> - Определяет сообщение, содержащее вызов инструмента.
    *   `context`: <begin>context и <end> - Определяет сообщение, содержащее дополнительный контекст.

    Кроме того, я использую специальные токены для вызова инструментов:
    *   `<tool_call>` и `</tool_call>`:  Обозначают вызов инструмента и содержат JSON-object с параметрами.

    Примеры:
    `<tool_call>` в Granite не имеет закрывающего токена и заменяется на специальный токен `<|tool_call|>` по таблице специальных токенов.
    `<begin>` в Qwen заменяется на `<|im_start|>`, а `<end>` заменяется на `<|im_end|>` по таблице специальных токенов.
    После токена роли `system` и `user` и `assistant` следует разделитель `:`. Может следовать метка времени в формате ISO 8601 UTC :
      - `YYYY-MM-DD`
      - `YYYY-MM-DDThh:mm<TZDSuffix>`
      - `YYYY-MM-DDThh:mm:ss<TZDSuffix>`

    или штамп времени в формате UNIX, например `[1746241500]`. 

    Токенизер BPE (byte-pair encoding) использует словарь токенов. В словаре некоторые слова начинаются с символа `Ġ` - пробел, разделитель слов, 
    могут сшиваться без разделителя или с разделителем. 

    Итак, имея под рукой словарь токенов, мы можем сохранять содержимое в бинарном виде. При восстановлении контекста из журнала, мы
    используем разметку принятую в для данной модели. Общие принципы - наличие открывающих и закрывающих токенов, разделитель слов и т.д.
    Так например, в модели Qwen токены `<begin>` и `<end>` используются для разделения сообщений. 


Unescape
\" : Escaped double quote.
\\ : Escaped backslash.
\/ : Escaped forward slash (optional but often used)
\b : Backspace.
\f : Form feed.
\n : Newline.
\r : Carriage return.
\t : Horizontal tab.


 */

#include <glib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "json.h"
#include "quarks.h"


typedef struct _Tool {
    char* name;
    char* description;
    char* parameters;
} Tool_t;
typedef struct _Tool_call {
    char* name;
    char* arguments;
    char* id;       //!< идентификатор вызова, откуда берется?
} Tool_call_t;
typedef struct _Message {
    char* role;
    char* content;
    char* reasoning_content;
    char* tool_name;
    char* tool_call_id;
    uint32_t utc;
    GPtrArray* content_parts;
    GPtrArray* tool_calls; 
} Message_t;

/*! \brief работа с шаблонами, подстановка строк
    ${name}
    ${name:3:2} -- подстрока от третьего символа длиной два символа

Выражение слов \see 3.5.3 Shell Parameter Expansion
${parameter}
	возвращает значение переменной
${parameter:-значение}
	Если значение переменной не определено, то возвращается «значение». Не изменяет значение переменной
${parameter:=значение}
	Если значение переменной не определено, присваивает переменной значение и возвращает значение.
${parameter:?[значение]}
${parameter:+значение}
	Если значение переменной определено, то возвращается «значение».
${#parameter}
	Длина строки
${parameter:позиция}
	выделяет подстроку начиная с заданной позиции
${parameter:позиция:длина}
	выделяет подстроку заданной длины, начиная с заданной позиции
${parameter#шаблон}
	Удаление префикса
${parameter##шаблон}
	Удаление наибольшего префикса
${parameter%шаблон}
	Удаление суффикса
${parameter%%шаблон}
	Удаление наибольшего суффикса

${parameter/pattern/string}
${parameter//pattern/string}
${parameter/#pattern/string}
${parameter/%pattern/string}

* <https://www.gnu.org/software/bash/manual/bash.html#Shell-Parameter-Expansion>
 */
struct _String {
    char*  str;  //!< строка
    size_t size; //!< размер строки в байтах
    size_t pos;  //!< позиция записи в строке
};
void _string_append_len (struct _String * str, const char* s, size_t len);
void _string_append_c   (struct _String * str, unsigned int ch);
static inline
void _string_append(struct _String * str, const char* s) {
    _string_append_len(str, s, __builtin_strlen(s));
}
static inline
void _string_truncate(struct _String * str, size_t len)
{
    if (str->pos > len) 
        str->pos = len;
}


#define _STRING_ALIGN 64 // выравнивание на 64 байта
/*! \brief добавление строки с заданным
 */
void _string_append_len(struct _String * str, const char* s, size_t len)
{
    if (str->pos+len > str->size) {// применить выравнивание на 64
        size_t size = (str->size+len+_STRING_ALIGN-1)%_STRING_ALIGN;
        str->str = g_realloc(str->str, size);
        str->size = size;
    }
    __builtin_memcpy(&str->str[str->pos], s, len);
    str->pos += len;
    str->str[str->pos] = '\0';
}
/*! \brief добавление символа к строке */
void _string_append_c(struct _String * str, unsigned int ch) {
    if (str->pos+2 > str->size) {
        size_t size = (str->size+2+_STRING_ALIGN-1)%_STRING_ALIGN;
        str->str = g_realloc(str->str, size);
        str->size= size;
    }
    // кодирование в utf8?
    str->str[str->pos++] = ch;
    str->str[str->pos] = '\0';
}
static void _string_append_printf(struct _String * str, const char* format, ...){
    va_list args;
    va_start(args, format);
    int len = __builtin_vsnprintf(&str->str[str->pos], str->size-str->pos, format, args);
    va_end(args);
    str->pos += len;
    str->str[str->pos] = '\0';
}

#define _LOG_DOMAIN "messages"
/*! \brief запись в журнал сообщений
 */
char* qnn_message_log(char* role, uint64_t utc, const char* content, size_t size){
    char* buf = g_malloc(size+1);
    // todo escape \" некоторые символы должны быть заменены
    g_log(_LOG_DOMAIN, G_LOG_LEVEL_MESSAGE, 
        "{\"role\": \"%s\", \"content\": \"%.*s\", \"timestamp\": %lu}", 
        role, (int)size, content, utc);
    return buf;
}
#if 0
struct _String * _string_new(const char* buf){
    size_t size = __builtin_strlen(buf);
    char* buffer = g_malloc(size+1);
    struct _String *str_buf = g_slice_new();// {buffer, size};
    str_buf->str = buffer;
    __builtin_memcpy(str_buf->str, buf, size);
    str_buf->pos = size;
    str_buf->str[str_buf->size] = '\0';
    return str_buf;
}
#endif
/*! \brief convert template 
	\param buf - буфер для записи строки результата
	\param vars - список переменных с поиском по идентификаторам Quark
	\param templ - шаблон поиска с разметкой bash-style или jinja-style
	\return длина записи в буфере
 */
size_t _query_template(char *buffer, GSList* vars, const char* templ)
{
    size_t size = 4096;//__builtin_strlen(buf);
    //__builtin_memcpy(buffer, buf, size);
    struct _String str_buf = {.str = buffer, .size = size, .pos = 0};
    struct _String * stri = &str_buf;
//    GString * stri = g_string_new(buf);
	const char* s = templ;
	const char* open_tag = "${";
	const char* close_tag = "}";
	while(s[0]!='\0'){
		const char* str = s;
		while(s[0]!='\0' && !(s[0]=='\\' && s[1]=='"') && !(s[0]==open_tag[0] && s[1]==open_tag[1]) 
                         && !(s[0]=='{'  && s[1]=='{') && !(s[0]=='{' && s[1]=='#')) s++;
		int len = s-str;// длина префикса
        _string_append_len(stri, str, len);
		if(s[0]=='\0') {
			break;
		} else
        if (s[0]=='\\' && s[1]=='"') {// unescape \\ \r \n \"
			char ch=0;
			switch (s[1]){
			case '"':  ch = '"';  break;
			case 'n':  ch = '\n'; break;
			case 'r':  ch = '\r'; break;
			case 't':  ch = '\t'; break;
			case '\\': ch = '\\'; break;
			case '0':  ch = '\0'; break; // \nnn base 8
//			case 'x':  ch =  ...; break; // \xnn base 16
			default:
				ch = s[0]; s-=1; break;
			}
            _string_append_c(stri, ch);
            s+=2;
            continue;// угадай, куда будет переход?
        } else
		if (s[0]=='$'  && s[1]=='{') close_tag = "}" ;	// bash-style  expression
        else {
            if (s[0]=='{' && s[1]=='{') close_tag = "}}"; 	// jinja-style expression
            if (s[0]=='{' && s[1]=='%') close_tag = "%}"; 	// jinja-style operator
            if (s[0]=='{' && s[1]=='#') close_tag = "#}"; 	// jinja-style comment
        }
		{ // начало подстановки
			s+=2;
			while(isspace(s[0])) s++;
			if (isalpha(s[0])){
				const char* var_name = s;
				s++;
				while(isalnum(s[0]) || s[0]=='_') s++;
				size_t var_len = s-var_name;// длина имени
                char varname[var_len+1];// выделяем место на стеке
                __builtin_memcpy(varname, var_name, var_len);
                varname[var_len] = '\0';
				GQuark id = g_quark_try_string(varname);
                if (s[0] =='.') { // method
                    s++;
                } else
                if (s[0] ==':') { // substring
                    s++;
                } else
                if (s[0] =='%') { // format substitution
                    s++;
                } else
                if (s[0] =='|') { // filter
                    s++;
                }
				if (id !=0) {
					GSList* list = json_get_object(vars, id);
					if (list!=NULL){
						JsonNode* node  = (JsonNode*)list->data;
						if (node->type==JSON_BOOL)
							_string_append_printf(stri, "%s", node->value.b?"true":"false");
						else
						if (node->type==JSON_INT)
							_string_append_printf(stri, "%lld", node->value.i);
						else
						if (node->type==JSON_UINT)
							_string_append_printf(stri, "%llu", node->value.u);
						else
						if (node->type==JSON_FLOAT)
							_string_append_printf(stri, "%f", node->value.f);
						else
						if (node->type==JSON_STRING) {
							_string_append_printf(stri, "%s", node->value.s);
						} else
						if (node->type==JSON_ARRAY) {
						} else {// тип не определен
						}
					} else {// параметр не найден
					}
				}
			// printf("var %.*s\n", (int)var_len, var_name);
			} else if (s[0]=='"') { // строка
			} else if (isdigit(*s)) {
				unsigned int idx = (*s++ -'0');
				while(isdigit(*s)) 
					idx = idx*10 + (*s++ -'0');
				// подставить аргумент функции по индексу
				
			} else {
			}
			while(isspace(s[0])) s++;
			if(s[0]==close_tag[0] && close_tag[1]==0){ s++; }
			else
			if(s[0]==close_tag[0] && s[1]==close_tag[1]){ s+=2; }
			else {// ошибка в шаблоне

			}
		}
	}
	return stri->pos;
}
/*! \brief Разбор ответа от модели, выделение блока json или tool_call
    \note после разбора следует проверить, следует ли закрывающий тэг
 */
JsonNode* qnn_query_tool_call(const char* open_tag, const char* close_tag, char* str, char** call, char** tail){
    char *s = str;
    char *start = strstr(s, open_tag);
    if (call !=NULL) *call = start;
    if (start==NULL) return NULL;
    s = start+strlen(open_tag);
    while (isspace(*s)) s++;
    // TODO запустить валидацию
    char *end = strstr(s, close_tag);
    JsonNode* node = json_value(s, tail, NULL);
    if (end!=NULL && tail!=NULL) 
        *tail = end+strlen(close_tag);
    return node;
}
/*! \brief Разбор строки, выражение слов в запросе с использованием шаблонов 
    Шаблоны make: $() операторы могут вызвать команды и вставлять результат в строку, например $(shell ...)
    $(function arguments)

    Шаблоны bash: ${}
    Шаблоны PHP: <? ?> <?php if ($expression == true): ?>
    Шаблоны разметки XML: <function_call> <function_result>
    Шаблоны jinja: {{- }}  или {%- %}

    \param quarks хеш таблица переменных окружения
    \param buffer буфер для результата
    \param size размер буфера
    \param str строка из запроса содержащая шаблоны подстановки
    \return строка с подставленными значениями
 */
static char* _query_eval(GData** vars, char* buffer, size_t size, const char* query)
{
    QTable_t* quarks = NULL;//ctx->quarks;
    char* s =  buffer;
    if (buffer==NULL) buffer = g_malloc(size);
    char close_tag = '}';
    struct _String str_buf = {buffer, size};
    struct _String * str = &str_buf;
    while (s[0]!='\0'){
        if (s[0]=='[' && s[1]=='{'){// шаблон json
            JsonNode* node = json_value(s, &s, NULL);
        } else
        if (s[0]=='[' && s[1]=='{'){// шаблон yaml
            JsonNode* node = json_value(s, &s, NULL);
        } else
        if (s[0]=='{' && s[1]=='%'){// шаблон jinja
        } else
        if (s[0]=='$' && s[1]=='('){// выделяем функцию
            s+=2;
            if (isdigit(s[0])){// $(1)..
                uint64_t val = (*s++ - '0');
                while (isdigit(s[0]))  // подставить аргумент
                    val = val*10 + (*s++ -'0');
            } else if (isalpha(s[0])) {// командной строки
                char* var_name = s++;
                while(isalnum(s[0]) || s[0]=='_') s++;
                int len = s-var_name;
                if (len == 5 && strncmp(var_name, "shell",  5)) {// выполнить команду на локальном компьютере
                } else 
                if (len == 3 && strncmp(var_name, "get",    3)) {// HTTP GET загрузить содержимое web страницы
                } else
                if (len == 4 && strncmp(var_name, "post",   4)) {// HTTP POST загрузить содержимое web страницы
                } else
                if (len == 6 && strncmp(var_name, "delete", 6)) {// HTTP DELETE удалить содержимое
                } else
                if (len == 6 && strncmp(var_name, "notify", 6)) {// HTTP NOTIFY отослать уведомление
                } else
                if (len == 6 && strncmp(var_name, "create", 6)) {// HTTP PUT создание объекта через запрос 
                } else {
                }
            } else if (s[0]=='(') {// evaluate expression
            } else {

            }
        } else
        if (s[0]=='$' && s[1]=='{'){// выделяем переменную
            if (s[1]=='(') close_tag == ')';
            else close_tag = '}';
            s+=2;
            char* var_name = s;
            while(s[0]!=close_tag && s[0]!=':' && s[0]!='/' && s[0]!='\0') s++;// выполняем поиск параметра с разделителем
            var_name = g_strndup(var_name, s - var_name);// alloca!
            unsigned id = _quark_lookup(quarks, var_name);

            char* value = g_datalist_get_data(vars, var_name);
            if (s[0]==':' && s[1]=='-') {// ${переменная:-значение} значение по умолчанию
                s+=2;
                char* default_value = s;
                while(s[0]!=close_tag && s[0]!='\0') s++;
                if (value==NULL || value[0]=='\0') {
                    _string_append_len(str, default_value, s- default_value);
                } else {
                    _string_append(str, value);
                }
            } else
            if (s[0]==':' && s[1]=='@') {// вставляет обратные косые для представления строки или пустой строки в форме NULL
                s+=2;
                char* default_value = s;
                while(s[0]!=close_tag && s[0]!='\0') s++;
                if (value==NULL || value[0]=='\0') {
                    _string_append_len(str, default_value, s- default_value);
                } else {
                    char* v = value;
                    _string_append_c(str, '"');
                    do {
                        int len=0;
                        while (v[len]!='"' && v[len]!='\\' && v[len]!='\0') len++;
                        if (len) {
                            _string_append_len(str, v, len);
                            v += len;
                        }
                        if (v[0]!='\0') {
                            _string_append_c(str, '\\');
                            // возможно стоит рассмотреть символы \n \t \r
                            _string_append_c(str, v[0]);
                            v++;
                        }
                    } while(v[0]!='\0');
                    _string_append_c(str, '"');
                }
            } else
            if (value) {
                int offset = (s[0]==':')?strtol(&s[1],&s,0):0;
                if (s[0]==':'){
                        int length = strtol(&s[1],&s,0);
                        char * subs = &value[offset];
                        if (!g_utf8_validate(subs,-1, NULL)){
                            g_print("!!!!! ACHTUNG invalid utf8 char at pos ?? !!!!\n");
                        }
                        int len = g_utf8_strlen(subs, -1);
                        char* end = g_utf8_offset_to_pointer (subs, MIN(len, length));//len>length?length:len);

                        _string_append_len(str, subs, end - subs);
                } else
                    _string_append(str, &value[offset]);
            }
            g_free(var_name);
            if(s[0]==close_tag) s++;
        } else {
            _string_append_c(str, s[0]);
            s++;
        }
    }
    str->str[str->pos] = '\0';
    return str_buf.str;
}

#define _JCONFIG_NAMESPACE \
    TAG(messages),TAG(model),TAG(tools),TAG(name),TAG(role),TAG(content),TAG(url)
// Перечисление идентификаторов полей в запросах и отвехат протокола Startum
#define TAG(name) #name // stringify
static const char* names[] = {// список имен идентификаторов
	_JCONFIG_NAMESPACE,
	NULL
};
#undef TAG
#define TAG(name) JCONFIG_TAG_##name // concatenate
enum _json_tags {// перечисление идентификаторов
	_JCONFIG_NAMESPACE,

	_TAG_COUNT
};

static GQuark tags[_TAG_COUNT];
static void __attribute__((constructor)) _tag_init()
{// статический конструктор, формирует таблицу идентификаторов
	for(int i=0; i<_TAG_COUNT; i++){
		tags[i] = g_quark_from_static_string(names[i]);
	}
}
#undef  TAG
#define TAG(name) tags[JCONFIG_TAG_##name]


/*! \brief заполнение промпта с использованием определения функций инструментов */
char* _query_json(char* query, JsonNode* json)
{
    const char* model = json_get_string(json->value.list, TAG(model), "Gemma3-27b");//"model": "llama3.1-70b",
    const char* role  = json_get_string(json->value.list, TAG(role), "user");
    GSList* tools     = json_get_array(json->value.list,  TAG(tools));
    GSList* messages  = json_get_array(json->value.list,  TAG(messages));
    GString* str = g_string_new(query);
    if (tools){
        g_string_append(str,"<|im_start|>system");
        g_string_append(str,"\n\n# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>");
        GSList* list = tools;
        while(list){
            JsonNode* tool = list->data;
            json_to_string(tool, str, 2);
            list = list->next;
        }
        g_string_append(str,"</tools>\n");
        g_string_append(str,"<|im_end|>\n");
    }
    if (messages){
        GSList* list = messages;
        while(list){
            JsonNode* message = list->data;
            const char* role  = json_get_string(message->value.list, TAG(role), "user");
            const char* content = json_get_string(message->value.list, TAG(content), ""); 
            g_string_append(str,"<|im_start|>");
            if (strcmp(role, "tool")==0){// ответ на вызов инструмента
                g_string_append(str,"user\n");
                g_string_append(str,"<tool_response>\n");
                g_string_append(str,content);
                g_string_append(str,"</tool_response>\n");
            } else {
                g_string_append(str,role);
                g_string_append_c(str,'\n');
                g_string_append(str,content); 
            }
            g_string_append(str,"<|im_end|>\n");
            list = list->next;
        }
    }

    return g_string_free(str, FALSE);

}