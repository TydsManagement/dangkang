import LlmSettingItems from '@/components/llm-setting-items';
import { variableEnabledFieldMap } from '@/constants/chat';
import {
  ModelVariableType,
  settledModelVariableMap,
} from '@/constants/knowledge';
import { useTranslate } from '@/hooks/commonHooks';
import { Variable } from '@/interfaces/database/chat';
import { Form, Input, Switch } from 'antd';
import { useCallback, useEffect } from 'react';
import { IOperatorForm } from '../interface';

const GenerateForm = ({ onValuesChange, form }: IOperatorForm) => {
  const { t } = useTranslate('flow');
  const initialLlmSetting = undefined;

  const handleParametersChange = useCallback(
    (value: ModelVariableType) => {
      const variable = settledModelVariableMap[value];
      form?.setFieldsValue(variable);
    },
    [form],
  );

  useEffect(() => {
    const switchBoxValues = Object.keys(variableEnabledFieldMap).reduce<
      Record<string, boolean>
    >((pre, field) => {
      pre[field] =
        initialLlmSetting === undefined
          ? true
          : !!initialLlmSetting[
              variableEnabledFieldMap[
                field as keyof typeof variableEnabledFieldMap
              ] as keyof Variable
            ];
      return pre;
    }, {});
    const otherValues = settledModelVariableMap[ModelVariableType.Precise];
    form?.setFieldsValue({ ...switchBoxValues, ...otherValues });
  }, [form, initialLlmSetting]);

  return (
    <Form
      name="basic"
      labelCol={{ span: 9 }}
      wrapperCol={{ span: 15 }}
      autoComplete="off"
      form={form}
      onValuesChange={onValuesChange}
    >
      <LlmSettingItems
        handleParametersChange={handleParametersChange}
      ></LlmSettingItems>
      <Form.Item
        name={['prompt']}
        label={t('prompt', { keyPrefix: 'knowledgeConfiguration' })}
        initialValue={t('promptText', { keyPrefix: 'knowledgeConfiguration' })}
        tooltip={t('promptTip', { keyPrefix: 'knowledgeConfiguration' })}
        rules={[
          {
            required: true,
            message: t('promptMessage'),
          },
        ]}
      >
        <Input.TextArea rows={8} />
      </Form.Item>
      <Form.Item
        name={['cite']}
        label={t('cite')}
        initialValue={true}
        valuePropName="checked"
        tooltip={t('citeTip')}
      >
        <Switch />
      </Form.Item>
    </Form>
  );
};

export default GenerateForm;
