### Public 繼承
```cpp
class B: public A
{
//類別內容 
};
```


| B的public | B的protected | B的private |
| :-------: | :----------: | :--------: |
| A的public | A的protected |     -      |

### Protected 繼承
```cpp
class B: protected A
{
//類別內容 
};
```


| B的public |     B的protected     | B的private |
| :-------: | :------------------: | :--------: |
|     -     | A的public和protected |     -      |

### Private 繼承
```cpp
class B: private A
{
//類別內容 
};
```


| B的public | B的protected |      B的private      |
| :-------: | :----------: | :------------------: |
|     -     |      -       | A的public和protected |

