#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <curl/curl.h>
#include <openssl/md5.h>
#include "htmlstreamparser.h"
 
#define MAXQSIZE 9000000       // Maximum size of the queue, q
#define MAXURL 100000          // Maximum size of a URL
#define MAXPAGESIZE 200000          // Maximum size of a webpage
#define MAXDOWNLOADS 2000      // Maximum number of downloads we will attempt


struct buffer{
    char *buf;
    size_t size;
};


struct node{
    char *data;
    struct node *left;
    struct node *right;
}*url_tree;


//prototypes
void AppendQueue(struct buffer *buf, char *weblinks);
int QueueSize(char *q);
int ShiftCursor(int c, char *q);
int GetNextURL(int c, char *q, char *myurl);

size_t curl_callback(char *buffer, size_t size, size_t nmemb, void * buff);
char *GetLinksFromWebPage(char *myhtmlpage, char *myurl);
char *GetWebPage(char *myurl);
int Whitelist(char *url);

char *str2md5(const char *str, int length);
char *md52str(const char *str, int length);
//tree
int find_in_tree(char *url);
void find_in_tree2(char *url, struct node **par, struct node **loc);
void insert_in_tree(char *url);


//not implemented
char *md52str(const char *str, int length) {
    return strdup(str);
}


char *str2md5(const char *str, int length) {
    int n;
    MD5_CTX c;
    unsigned char digest[16];
    char *out = (char*)malloc(33);

    MD5_Init(&c);

    while (length > 0) {
        if (length > 512) {
            MD5_Update(&c, str, 512);
        } else {
            MD5_Update(&c, str, length);
        }
        length -= 512;
        str += 512;
    }

    MD5_Final(digest, &c);

    for (n = 0; n < 16; ++n) {
        snprintf(&(out[n*2]), 16*2, "%02x", (unsigned int)digest[n]);
    }
	return out;
}


size_t curl_callback(char *buffer, size_t size, size_t nmemb, void * buff) {
    if (size == 0)
        return 0;
    struct buffer * buf = (struct buffer *) buff;
    size_t rsize = nmemb * size;
    void * temp = realloc(buf->buf, buf->size + rsize + 1);
    if (temp == NULL)
        return 0;
    buf->buf = temp;
    memcpy(buf->buf + buf->size, buffer, rsize);
    buf->size += rsize;
    buf->buf[buf->size] = 0;
    return rsize;
}


char *GetWebPage(char *myurl) {
    struct buffer mybuf = {0};
    CURL *curl = curl_easy_init();
    curl_easy_setopt( curl, CURLOPT_URL, myurl);
    curl_easy_setopt( curl, CURLOPT_WRITEFUNCTION, curl_callback);
    curl_easy_setopt( curl, CURLOPT_WRITEDATA, &mybuf);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1);

    //Download the webpage
    CURLcode curl_res = curl_easy_perform(curl);

    curl_easy_cleanup(curl);

    if (curl_res == 0)
        return mybuf.buf;
    else
        return NULL;
}


char *GetTitleFromWebPage(char *html) {
    HTMLSTREAMPARSER *hsp = html_parser_init();

    //htmlparser setup
    char tag[6], *title, inner[128];
    size_t title_len = 0;

    size_t size = strlen(html);
    size_t nmemb = sizeof(char);
	size_t realsize = size * nmemb, p; 

    html_parser_set_tag_buffer(hsp, tag, sizeof(tag));
    html_parser_set_inner_text_buffer(hsp, inner, sizeof(inner)-1);

    title = malloc(MAXPAGESIZE * sizeof(char));
    title[0] = '\0';

    for (p = 0; p < realsize; p++){
        html_parser_char_parse(hsp, ((char *)html)[p]);
        if (html_parser_cmp_tag(hsp, "/title", 6)) {
            title_len = html_parser_inner_text_length(hsp);
            title = html_parser_replace_spaces(html_parser_trim(html_parser_inner_text(hsp), &title_len), &title_len);
            break; 
        }
    }
    html_parser_cleanup(hsp);
    if (title_len > 0) {
        title[title_len] = '\0';
        return strdup(title);
    }
    return NULL;
}



char *ParseTitle(char *t_1){
    char *t_2;
    int i, j;
    t_2 = (char *)malloc(MAXPAGESIZE * sizeof(char));

    i = 0;
    j = 0;
    while (i < strlen(t_1)){
        if (t_1[i] == ','
            || t_1[i] == '.'
            || t_1[i] == ':'
            || t_1[i] == '"'
            || t_1[i] == '?'
            || t_1[i] == '!'
            || t_1[i] == ';'){
            t_2[j++] = ' ';
        } else if (t_1[i] >= '0' && t_1[i] <= '9'){
            while (t_1[i] >= '0' && t_1[i] <= '9'){
                i++;
            }
            strcpy(&t_2[j], "<X>");
            j += 3;
        } else if (t_1[i] >= 'A' && t_1 <= 'Z'){
            t_2[j++] = t_1[i++] - ('A' - 'a');
        } else {
            t_2[j++] = t_1[i++];
        }

    }
    t_2[j] = '\0';
    strcat(t_2, " <EOS>\n");
    return strdup(t_2);
}


char *GetLinksFromWebPage(char *html, char *url) {
    HTMLSTREAMPARSER *hsp = html_parser_init();

    //htmlparser setup
    char tag[2];
    char attr[5];
    char val[128];
    html_parser_set_tag_to_lower(hsp, 1);
    html_parser_set_attr_to_lower(hsp, 1);
    html_parser_set_tag_buffer(hsp, tag, sizeof(tag));
    html_parser_set_attr_buffer(hsp, attr, sizeof(attr));
    html_parser_set_val_buffer(hsp, val, sizeof(val)-1);
    
    char *urllist = malloc(MAXPAGESIZE * sizeof(char));
    urllist[0] = '\0';
    size_t realsize = strlen(html), p;

    for (p = 0; p < realsize; p++)
    {
        html_parser_char_parse(hsp, ((char *)html)[p]);
        if (html_parser_cmp_tag(hsp, "a", 1))
            if (html_parser_cmp_attr(hsp, "href", 4))
                if (html_parser_is_in(hsp, HTML_VALUE_ENDED))
                {
                    html_parser_val(hsp)[html_parser_val_length(hsp)] = '\0';
                    if (strstr(html_parser_val(hsp), "http") == NULL){
                        strcat(urllist, url);
                    }
                    if (strlen(urllist) + strlen(html_parser_val(hsp)) > MAXPAGESIZE){
                        html_parser_cleanup(hsp);
                        return urllist;
                    }
                    strcat(urllist, html_parser_val(hsp));
                    strcat(urllist, "\n");
                }
    }

    html_parser_cleanup(hsp);

    return urllist;
}


int QueueSize(char *q) {
    int k, total;
    total = 0;
    for(k = 0; k < MAXQSIZE; k++) {
        if(q[k] == '\n') {
            total++;
        }
        if(q[k] == '\0') {
            return total;
        }
    }
    return total;
}


void AppendSeq(struct buffer *buf, char *title) {
    size_t rsize = strlen(title) * sizeof(char);
    void *temp = realloc(buf->buf, buf->size + rsize + 1);
    if (temp == NULL){
        printf("RIP memory\n...Exiting\n");
    } else {
        buf->buf = temp;
        memcpy(buf->buf + buf->size, title, rsize);
        buf->size += rsize;
        buf->buf[buf->size] = '\0';
    }
}


void AppendURL(struct buffer *buf, char *weblinks) {
    char url[MAXURL], *url_md5;
    int i, r;
    r = 0;
    for (i = 0; i < QueueSize(weblinks); i++){
        GetNextURL(r, weblinks, url);
        r = ShiftCursor(r, weblinks);
        if (Whitelist(url)){
            url_md5 = str2md5(url, strlen(url));
            if (!find_in_tree(url_md5)) {
                insert_in_tree(url);
                size_t rsize = strlen(url) * sizeof(char);
                void *temp = realloc(buf->buf, buf->size + rsize + 2);
                if (temp == NULL){
                    printf("RIP memory\n...Exiting\n");
                } else {
                    buf->buf = temp;
                    memcpy(buf->buf + buf->size, url, rsize);
                    buf->size += rsize;
                    buf->buf[buf->size++] = '\n';
                    buf->buf[buf->size] = '\0';
                }
            } 
        } 
    }
}


int GetNextURL(int c, char *q, char *url) {
    char *p = &q[c]; 
    int i;
    for(i = 0; i < MAXURL; i++) {
        if(p[i] == '\n') {
            url[i] = '\0';
            return 1;
        } else {
            url[i] = p[i];
        }
    }
    strcpy(url,"http://127.0.0.1");
    return 0;
}


int ShiftCursor(int c, char *q) {
    char *p = &q[c];
    int k;
    for (k = 0; k < MAXURL; k++){
        if(p[k] == '\n'){
            return c + k + 1;
        }
    }
    return 0;
}


void find_in_tree2(char *url, struct node **par, struct node **loc)
{
    *par = NULL;
    *loc = NULL;
    struct node *ptr,*ptrsave;
    if(url_tree == NULL) 
        return;
    if(!(strcmp(url, url_tree->data)))
    {
        *loc = url_tree;
        return;
    }
    ptrsave = NULL;
    ptr = url_tree;
    while(ptr != NULL) {
        if(!(strcmp(url, ptr->data))) break;
        ptrsave = ptr;
        if(strcmp(url, ptr->data) < 0)
            ptr = ptr->left;
        else
            ptr = ptr->right;
    }
    *loc = ptr;
    *par = ptrsave;
}


int find_in_tree(char *url)
{
    struct node *parent, *location;
    find_in_tree2(url, &parent, &location);
    if (location != NULL)
        return 1;
    else
        return 0;
}


void insert_in_tree(char *url) {
    struct node *parent, *location, *temp;
    find_in_tree2(url, &parent, &location);
    if(location != NULL)
    {
        return;
    }
    temp = malloc(sizeof(struct node));
    temp->data = malloc(MAXURL * sizeof(char));
    strcpy(temp->data, url);
    temp->left = NULL;
    temp->right = NULL;
    if(parent == NULL)
        url_tree = temp;
    else
        if(strcmp(url, parent->data) < 0)
            parent->left = temp;
        else
            parent->right = temp;
}


void WriteToFile(struct buffer *buf, char *path){
    FILE *f_out;
    f_out = fopen(path, "w");
    fprintf(f_out, "%s", buf->buf);
    fclose(f_out);
}

//simple whitelist function to add allowed sites more easily
int Whitelist(char *url) {
    const int wlist_size = 3;
    const char *wlist[] = {"leidenuniv.nl",
                                "liacs.nl",
                                "universiteitleiden.nl"};
    int i;
    for (i = 0; i < wlist_size; i++)
    {
        if (strstr(url, wlist[i]) != NULL)
            return 1;
    }
    return 0;
}


int main(int argc, char* argv[]) {
    char *url;
    char *html, *weblinks, *title;
    int k, qs, ql;
    int v;
    int p_1 = 0;


    //init queues
    struct buffer q_1 = {0};
    struct buffer q_2 = {0};
    url = (char *)malloc(MAXURL * sizeof(char));

    if (argc <= 1) {
        printf("\n\nNo webpage given...exiting\n\n"); 
        exit(0);
    } else { 
        strcpy(url, argv[1]);
        if(strstr(url,"http") != NULL) {
            printf("\nInitial web URL: %s\n\n", url);
        } else {
            printf("\n\nYou must start the URL with lowercase http...exiting\n\n"); 
            exit(0);
        }
    }

    if (!Whitelist(url)){
        printf("Not allowed...exiting...\n");
        exit(0);
    } else {
        strcat(url, "\n");
        AppendURL(&q_1, url);
    }

    for(k = 0; k < MAXDOWNLOADS; k++) {
        qs = QueueSize(q_1.buf); 
        ql = q_1.size;
        printf("\nDownload #: %d   Weblinks: %d   Queue Size: %d\n",k, qs, ql);

        if (!GetNextURL(p_1, q_1.buf, url)) {
            printf("\n\nNo URL in queue\n\n");
            exit(0);
        }
        p_1 = ShiftCursor(p_1, q_1.buf);

        if (Whitelist(url)) {
            printf("url=%s\n", url);
            html = GetWebPage(url);

            if (html == NULL) { 
                printf("\n\nhtml is NULL\n\n");
            } 

            if (html) { 
                v = strlen(html); 
                printf("\n\nwebpage size of %s is %d\n\n", url, v);
                weblinks = GetLinksFromWebPage(html, url);
                AppendURL(&q_1, weblinks);
                title = GetTitleFromWebPage(html);
                if (title != NULL){
                    title = ParseTitle(title);
                    AppendSeq(&q_2, title);
                }
            }
        } else {
            printf("\n\nNot in allowed domains: %s\n\n",url);
        }
        WriteToFile(&q_2, "data1.txt");
    }
    free(q_1.buf);
    free(q_2.buf);
    return 0;
}
